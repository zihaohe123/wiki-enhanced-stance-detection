import os
import socket

if __name__ == '__main__':
    data = ['vast', 'pstance', 'covid'][1]
    topic = ['bernie,biden', 'bernie,trump', 'biden,bernie', 'biden,trump', 'trump,bernie', 'trump,biden'][1]
    batch_size = 32
    epochs = 100
    patience = 10
    lr = 2e-5
    l2_reg = 5e-5
    model = ['bert-base', 'bertweet', 'covid-twitter-bert'][1]
    wiki_model = ['', 'bert-base'][1]
    n_layers_freeze = 0
    n_layers_freeze_wiki = 10
    gpu = '0,1'

    if wiki_model == model:
        n_layers_freeze_wiki = n_layers_freeze
    if not wiki_model or wiki_model == model:
        n_layers_freeze_wiki = 0

    os.makedirs('results', exist_ok=True)
    if data != 'vast':
        file_name = f'results/{data}-topic={topic}-lr={lr}-bs={batch_size}.txt'
    else:
        file_name = f'results/{data}-lr={lr}-bs={batch_size}.txt'

    if model != 'bert-base':
        file_name = file_name[:-4] + f'-{model}.txt'
    if n_layers_freeze > 0:
        file_name = file_name[:-4] + f'-n_layers_fz={n_layers_freeze}.txt'
    if wiki_model:
        file_name = file_name[:-4] + f'-wiki={wiki_model}.txt'
    if n_layers_freeze_wiki > 0:
        file_name = file_name[:-4] + f'-n_layers_fz_wiki={n_layers_freeze_wiki}.txt'

    n_gpus = len(gpu.split(','))
    file_name = file_name[:-4] + f'-n_gpus={n_gpus}.txt'

    command = f"python3 -u src/train.py " \
              f"--data={data} " \
              f"--topic={topic} " \
              f"--model={model} " \
              f"--wiki_model={wiki_model} " \
              f"--n_layers_freeze={n_layers_freeze} " \
              f"--n_layers_freeze_wiki={n_layers_freeze_wiki} " \
              f"--batch_size={batch_size} " \
              f"--epochs={epochs} " \
              f"--patience={patience} " \
              f"--lr={lr} " \
              f"--l2_reg={l2_reg} " \
              f"--gpu={gpu}" \
              # f" > {file_name}"

    print(command)
    os.system(command)