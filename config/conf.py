import os
import json
import argparse


def path_config(data_path):
    assert os.path.exists(data_path)
    with open(data_path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    print(opts)
    return opts


def arg_config():
    parse = argparse.ArgumentParser('text-level gnn')
    parse.add_argument('--cuda', type=int, default=-1, help='cuda device, default cpu')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate of training')
    parse.add_argument('-bt1', '--beta1', type=float, default=0.9, help='beta1 of Adam optimizer 0.9')
    parse.add_argument('-bt2', '--beta2', type=float, default=0.98, help='beta2 of Adam optimizer 0.999')
    parse.add_argument('-eps', '--eps', type=float, default=1e-9, help='eps of Adam optimizer 1e-8')
    parse.add_argument('-warmup', '--warmup_step', type=int, default=10000, help='warm up steps for optimizer')
    parse.add_argument('--decay', type=float, default=0.95, help='lr decay rate for optimizer')
    parse.add_argument('--decay_step', type=int, default=10000, help='lr decay steps for optimizer')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for Adam optimizer')
    parse.add_argument('--scheduler', choices=['cosine', 'inv_sqrt', 'exponent', 'linear', 'step', 'const'],
                       default='const', help='the type of lr scheduler')
    parse.add_argument('--grad_clip', type=float, default=5., help='the max norm of gradient clip')
    parse.add_argument('--max_step', type=int, default=50000, help='the total steps of training')
    parse.add_argument('--patient', type=int, default=10, help='patient number in early stopping')

    parse.add_argument('--batch_size', type=int, default=32, help='batch size of source inputs')
    parse.add_argument('--epoch', type=int, default=20, help='number of training')
    parse.add_argument('--update_step', type=int, default=1, help='gradient accumulation and update per x steps')
    parse.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
    parse.add_argument("--bert_lr", type=float, default=2e-5, help='bert learning rate')
    parse.add_argument("--bert_layers", type=int, default=4, help='the number of last bert layers')
    parse.add_argument('--bert_embed_dim', type=int, default=400, help='feature size of bert inputs')
    parse.add_argument('--wd_embed_dim', type=int, default=100, help='feature size of word')
    parse.add_argument('--tag_embed_dim', type=int, default=50, help='feature size of pos-tagging inputs')
    parse.add_argument('--char_embed_dim', type=int, default=100, help='feature size of char inputs')

    parse.add_argument('--hidden_size', type=int, default=128, help='feature size of hidden layer')
    parse.add_argument('--char_hidden_size', type=int, default=50, help='feature size of char-encoding hidden layer')
    parse.add_argument('--dec_hidden_size', type=int, default=100, help='feature size of decoder hidden layer')
    parse.add_argument('--rnn_depth', type=int, default=2, help='number of rnn layers')
    parse.add_argument('--enc_bidi', action='store_true', default=True, help='is encoder bidirectional?')

    parse.add_argument('-mpe', '--max_pos', default=600, help='max sequence position embeddings')
    parse.add_argument('--att_drop', type=float, default=0.1, help='attention dropout')
    parse.add_argument('--embed_drop', type=float, default=0.2, help='drop rate of embedding layer')
    parse.add_argument('--rnn_drop', type=float, default=0.2, help='drop rate of rnn layer')
    parse.add_argument('--dropout', type=float, default=0.33, help='dropout ratio')

    parse.add_argument('--model_chkp', type=str, default='model.pkl', help='model saving path')
    parse.add_argument('--vocab_chkp', type=str, default='vocab.pkl', help='vocab saving path')

    args = parse.parse_args()

    print(vars(args))

    return args



