from engine import Engine

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=('vast', 'pstance', 'covid'), default='vast',
                        help='which dataset to use')
    parser.add_argument('--topic', type=str, choices=('bernie', 'biden', 'trump',
                                                      'bernie,biden', 'bernie,trump',
                                                      'biden,bernie', 'biden,trump',
                                                      'trump,bernie', 'trump,biden',
                                                      'face_masks', 'fauci',
                                                      'stay_at_home_orders', 'school_closures', ''), default='bernie',
                        help='the topic to use')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2_reg', type=float, default=5e-5)
    parser.add_argument('--max_grad', type=float, default=0)
    parser.add_argument('--n_layers_freeze', type=int, default=0)
    parser.add_argument('--model', type=str, choices=('bert-base', 'bertweet', 'covid-twitter-bert'),
                        default='bert-base')
    parser.add_argument('--wiki_model', type=str, choices=('', 'bert-base'), default='')
    parser.add_argument('--n_layers_freeze_wiki', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--inference', type=int, default=0, help='if doing inference or not')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    print(args)

    engine = Engine(args)
    engine.train()