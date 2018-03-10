import os
from utils import ToxicTestLoader
import argparse
import logging
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from models.rnn_att import RnnAtt
from models.attention import SimpleAtt, SelfAtt
import utils
from utils import ToxicTestLoader
import preprocess


_train_file = 'dataset/train.npz'
_test_file = 'dataset/test.npz'
_n_classes = len(preprocess.y_names)

LOGLEVEL = (('debug', logging.DEBUG),
            ('info', logging.INFO),
            ('warn', logging.WARN),
            ('error', logging.ERROR))
LOG = logging.getLogger(__name__)

tqdm.monitor_interval = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--model', default='gru',
                        choices= ['gru', 'lstm', 'simple', 'selfatt'])
    parser.add_argument('--rnn-hidden-dim', type=int, default=512)
    parser.add_argument('--rnn-attention', action='store_true')
    parser.add_argument('--text-len', type=int, default=128)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--cv-path', help='Directory contains CV models')
    parser.add_argument('--model-file', help='Best single model file')
    parser.add_argument('--submit-file', default='dataset/submit.csv.gz')
    parser.add_argument('--validate', action='store_true',
                        help='Use train file for validation')
    parser.add_argument('--loglevel', default='info',
                        choices=[x[0] for x in utils.LOGLEVEL])
    return parser.parse_args()


def get_embedding_info(args):
    embedding_npz = np.load(utils._embedding_file)
    # fill embedding dimension
    embedding_list = embedding_npz['embedding_list']
    args.embed_dim = len(embedding_list[0])
    # check sequence length
    train_embedding = embedding_npz['train_embedding']
    text_len = len(train_embedding[0])
    assert(args.text_len <= text_len)

    LOG.info('embedding dimension = {}, text length = {}({})'.format(
        args.embed_dim, args.text_len, text_len))


def get_samples(args):
    if args.validate:
        y_true = np.load(_train_file)['y']
        id = None
    else:
        id = np.load(_test_file)['id']
        y_true = None

    return y_true, id


def get_model(args):
    if args.model == 'gru' or args.model == 'lstm':
        model = RnnAtt(args.embed_dim, args.rnn_hidden_dim,
                       args.text_len, _n_classes, model=args.model,
                       bidir=args.bidirectional, atten=args.rnn_attention,
                       cuda=args.cuda)
    elif args.model == 'simple':
        model = SimpleAtt(args.embed_dim, args.text_len, _n_classes,
                          cuda=args.cuda)
    elif args.model == 'selfatt':
        model = SelfAtt(args.embed_dim, args.text_len, _n_classes,
                        cuda=args.cuda)
    else:
        raise ValueError

    if args.cuda:
        model.cuda()

    return model


def show_validate(y_true, y_pred):
    loss = 0.0
    for c in range(y_true.shape[1]):
        loss += log_loss(y_true[:, c], y_pred[:, c], eps=1e-7)
    loss /= y_true.shape[1]
    auc = roc_auc_score(y_true, y_pred)
    print('loss: {:.4f}, auc: {:.4f}'.format(loss, auc))


def predict1(args, model, loader, n_samples):
    model.eval()

    y_pred = np.zeros((n_samples, _n_classes), dtype=np.float32)
    y_cnt = 0

    for text, X, _ in tqdm(loader):
        text = text[:, :args.text_len, :]
        text = Variable(text, requires_grad=False)
        X = Variable(X, requires_grad=False)
        if args.cuda:
            text = text.contiguous().cuda(async=True)
            X = X.cuda(async=True)

        predict = F.sigmoid(model(text, X))

        y_pred[y_cnt:y_cnt+X.size(0)] = predict.data.cpu().numpy()
        y_cnt += X.size(0)

    assert(y_cnt == n_samples)
    return y_pred


def predict(args):
    if args.cv_path and args.model_file:
        raise ValueError('Cannot set both "cv-path"and "model-file"')
    elif args.cv_path is None and args.model_file is None:
        raise ValueError('Either "cv-path" or "model-file" must be set')

    loader = ToxicTestLoader(args.batch_size, 4, validate=args.validate)()
    n_samples = len(loader.dataset)
    y_true, id = get_samples(args)

    if args.cv_path:
        model_files = []
        for f in os.listdir(args.cv_path):
            if f.endswith('.pt'):
                f = os.path.join(args.cv_path, f)
                if os.path.isfile(f):
                    model_files.append(f)
        model_files = sorted(model_files)
        assert model_files
        LOG.info('Found {} models: {}'.format(
            len(model_files), list(map(os.path.basename, model_files))))

        y_pred = np.zeros((n_samples, _n_classes), dtype=np.float32)
        model = get_model(args)
        for model_file in model_files:
            model.load_state_dict(torch.load(model_file))
            y_pred1 = predict1(args, model, loader, n_samples)
            if args.validate:
                show_validate(y_true, y_pred1)
            y_pred += y_pred1
        y_pred /= len(model_files)
    else:
        model = get_model(args)
        model.load_state_dict(torch.load(args.model_file))
        y_pred = predict1(args, model, loader, n_samples)
        if args.validate:
            show_validate(y_true, y_pred)

    if not args.validate:
        submit = pd.DataFrame(columns=preprocess.y_names, data=y_pred)
        submit.insert(0, 'id', id)
        compression = 'gzip' if args.submit_file.endswith('.gz') else None
        submit.to_csv(args.submit_file, index=False, float_format='%.9f',
                      compression=compression)
        print('{} ready to submit!'.format(args.submit_file))


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level = dict(LOGLEVEL)[args.loglevel])

    args.cuda = torch.cuda.is_available()
    LOG.debug('Train on {}'.format('GPU' if args.cuda else 'CPU'))

    get_embedding_info(args)
    predict(args)
