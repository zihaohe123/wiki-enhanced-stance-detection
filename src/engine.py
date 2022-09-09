import torch
import torch.nn as nn
import os
import numpy as np
from datasets import data_loader
from models import BERTSeqClf


class Engine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        print('Preparing data....')
        if args.inference == 0:
            print('Training data....')
            train_loader = data_loader(args.data, 'train', args.topic, args.batch_size, model=args.model,
                                       wiki_model=args.wiki_model, n_workers=args.n_workers)
            print('Val data....')
            val_loader = data_loader(args.data, 'val', args.topic, 2*args.batch_size, model=args.model,
                                     wiki_model=args.wiki_model, n_workers=args.n_workers)
        else:
            train_loader = None
            val_loader = None
        print('Test data....')
        test_loader = data_loader(args.data, 'test', args.topic, 2*args.batch_size, model=args.model,
                                  wiki_model=args.wiki_model, n_workers=args.n_workers)
        print('Done\n')

        print('Initializing model....')
        num_labels = 2 if args.data == 'pstance' else 3
        model = BERTSeqClf(num_labels=num_labels, model=args.model, n_layers_freeze=args.n_layers_freeze,
                           wiki_model=args.wiki_model, n_layers_freeze_wiki=args.n_layers_freeze_wiki)
        model = nn.DataParallel(model)
        if args.inference == 1:
            if args.data != 'vast':
                model_name = f"ckp/model_{args.data}.pt"
            else:
                model_name = f"ckp/model_{args.data}_{args.topic}.pt"
            print('\nLoading checkpoint....')
            state_dict = torch.load(model_name, map_location='cpu')
            model.load_state_dict(state_dict)
            print('Done\n')
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        criterion = nn.CrossEntropyLoss(ignore_index=3)

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args

    def train(self):
        if self.args.inference == 0:
            import copy
            best_epoch = 0
            best_epoch_f1 = 0
            best_state_dict = copy.deepcopy(self.model.state_dict())
            for epoch in range(self.args.epochs):
                print(f"{'*' * 30}Epoch: {epoch + 1}{'*' * 30}")
                loss = self.train_epoch()
                f1, f1_favor, f1_against, f1_neutral = self.eval('val')
                if f1 > best_epoch_f1:
                    best_epoch = epoch
                    best_epoch_f1 = f1
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                print(f'Epoch: {epoch+1}\tTrain Loss: {loss:.3f}\tVal F1: {f1:.3f}\n'
                      f'Val F1_favor: {f1_favor:.3f}\tVal F1_against: {f1_against:.3f}\tVal F1_Neutral: {f1_neutral:.3f}\n'
                      f'Best Epoch: {best_epoch+1}\tBest Epoch Val F1: {best_epoch_f1:.3f}\n')
                if epoch - best_epoch >= self.args.patience:
                    break

            print('Saving the best checkpoint....')
            self.model.load_state_dict(best_state_dict)
            if self.args.data != 'vast':
                model_name = f"ckp/model_{self.args.data}.pt"
            else:
                model_name = f"ckp/model_{self.args.data}_{self.args.topic}.pt"
            torch.save(best_state_dict, model_name)

        print('Inference...')
        if self.args.data != 'vast':
            f1_avg, f1_favor, f1_against, f1_neutral = self.eval('test')
            print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
                  f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}')
        else:
            f1_avg, f1_favor, f1_against, f1_neutral, \
            f1_avg_few, f1_favor_few, f1_against_few, f1_neutral_few, \
            f1_avg_zero, f1_favor_zero, f1_against_zero, f1_neutral_zero, = self.eval('test')
            print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
                  f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}\n'
                  f'Test F1_Few: {f1_avg_few:.3f}\tTest F1_Favor_Few: {f1_favor_few:.3f}\t'
                  f'Test F1_Against_Few: {f1_against_few:.3f}\tTest F1_Neutral_Few: {f1_neutral_few:.3f}\n'
                  f'Test F1_Zero: {f1_avg_zero:.3f}\tTest F1_Favor_Zero: {f1_favor_zero:.3f}\t'
                  f'Test F1_Against_Zero: {f1_against_zero:.3f}\tTest F1_Neutral_Zero: {f1_neutral_zero:.3f}')

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            stances = batch['stances'].to(self.device)
            if self.args.wiki_model and self.args.wiki_model != self.args.model:
                input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
            else:
                input_ids_wiki = None
                attention_mask_wiki = None

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                input_ids_wiki=input_ids_wiki, attention_mask_wiki=attention_mask_wiki)
            loss = self.criterion(logits, stances)
            loss.backward()
            self.optimizer.step()

            interval = max(len(self.train_loader)//10, 1)
            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f'Batch: {i+1}/{len(self.train_loader)}\tLoss:{loss.item():.3f}')

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def eval(self, phase='val'):
        self.model.eval()
        y_pred = []
        y_true = []
        mask_few_shot = []
        val_loader = self.val_loader if phase == 'val' else self.test_loader
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['stances']
                if self.args.data == 'vast' and phase == 'test':
                    mask_few_shot_ = batch['few_shot']
                else:
                    mask_few_shot_ = torch.tensor([0])
                if self.args.wiki_model and self.args.wiki_model != self.args.model:
                    input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                    attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
                else:
                    input_ids_wiki = None
                    attention_mask_wiki = None
                logits = self.model(input_ids, attention_mask, token_type_ids,
                                    input_ids_wiki=input_ids_wiki, attention_mask_wiki=attention_mask_wiki)
                preds = logits.argmax(dim=1)
                y_pred.append(preds.detach().to('cpu').numpy())
                y_true.append(labels.detach().to('cpu').numpy())
                mask_few_shot.append(mask_few_shot_.detach().to('cpu').numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        mask_few_shot = np.concatenate(mask_few_shot)

        from sklearn.metrics import f1_score
        if self.args.data != 'pstance':
            f1_against, f1_favor, f1_neutral = f1_score(y_true, y_pred, average=None)
        else:
            f1_against, f1_favor = f1_score(y_true, y_pred, average=None)
            f1_neutral = 0

        if self.args.data == 'pstance':
            f1_avg = 0.5 * (f1_favor + f1_against)
        else:
            f1_avg = (f1_favor + f1_against + f1_neutral) / 3

        if self.args.data == 'vast' and phase == 'test':
            mask_few_shot = mask_few_shot.astype(bool)
            y_true_few = y_true[mask_few_shot]
            y_pred_few = y_pred[mask_few_shot]
            f1_against_few, f1_favor_few, f1_neutral_few = f1_score(y_true_few, y_pred_few, average=None)
            f1_avg_few = (f1_against_few + f1_favor_few + f1_neutral_few) / 3

            mask_zero_shot = ~mask_few_shot
            y_true_zero = y_true[mask_zero_shot]
            y_pred_zero = y_pred[mask_zero_shot]
            f1_against_zero, f1_favor_zero, f1_neutral_zero = f1_score(y_true_zero, y_pred_zero, average=None)
            f1_avg_zero = (f1_against_zero + f1_favor_zero + f1_neutral_zero) / 3

            return f1_avg, f1_favor, f1_against, f1_neutral, \
                   f1_avg_few, f1_favor_few, f1_against_few, f1_neutral_few, \
                   f1_avg_zero, f1_favor_zero, f1_against_zero, f1_neutral_zero,

        return f1_avg, f1_favor, f1_against, f1_neutral
