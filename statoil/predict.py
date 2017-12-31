from __future__ import print_function
import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import log_loss
import pandas as pd

from models.simplenet import SimpleNet
from models.densenet import DenseNet
from utils import StatoilTestLoader, LOGLEVEL


LOG = logging.getLogger(__name__)

tqdm.monitor_interval = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle statoil competition')
    parser.add_argument('--test-file', default='dataset/test.npz')
    parser.add_argument('--validate', action='store_true',
                        help='Pass train file to --test-file for validation')
    parser.add_argument('--cv-path', help='Directory contains CV models')
    parser.add_argument('--model-file', help='Best single model file')
    parser.add_argument('--loglevel', default='info',
                        choices=[x[0] for x in LOGLEVEL])
    parser.add_argument('--cpu', action='store_true', help='Predict on CPU')
    parser.add_argument('--batch-size', type=int, metavar='N', default=128)
    parser.add_argument('--model', default='simplenet', choices=
                        ['densenet', 'simplenet'])
    parser.add_argument('--augment', action='store_true', help='Test augment')
    parser.add_argument('--submit-file', default='dataset/submit.csv')
    return parser.parse_args()


def get_model(args):
    if args.model == 'simplenet':
        model = SimpleNet()
    elif args.model == 'densenet':
        model = DenseNet(block_lst=(4, 4, 4, 4))
    else:
        raise ValueError

    if args.cuda:
        model.cuda()

    return model


def get_samples(args):
    f = np.load(args.test_file)

    if args.validate:
        y_true = f['label']
        ID = None
        n_samples = y_true.shape[0]
    else:
        ID = f['ID']
        y_true = None
        n_samples = ID.shape[0]

    LOG.info('Get {} test samples'.format(n_samples))
    return n_samples, y_true, ID


def show_validate(y_true, y_pred):
    loss = log_loss(y_true, y_pred)
    acc = ((y_pred > 0.5).astype(np.float32) == y_true).sum()
    acc /= len(y_true)
    print('loss: {:.4f}, acc: {:.0f}%'.format(loss, acc*100))


def predict_epoch(args, model, loader, n_samples):
    model.eval()

    n = 0
    y_pred = np.empty(n_samples)

    for X_img, _, _ in tqdm(loader):
        X_img = Variable(X_img, volatile=True, requires_grad=False)
        if args.cuda:
            X_img = X_img.cuda(async=True)
        predict = F.sigmoid(model(X_img))
        predict = predict.data.cpu().numpy()

        sz = X_img.size(0)
        y_pred[n:n+sz] = predict.squeeze()
        n += sz

    assert(n == n_samples)
    return y_pred


def predict(args):
    if args.cv_path and args.model_file:
        raise ValueError('Cannot set both "cv-path"and "model-file"')
    elif args.cv_path is None and args.model_file is None:
        raise ValueError('Either "cv-path" or "model-file" must be set')

    w_wo = 'w/' if args.augment else 'w/o'
    LOG.info('Predicting by {}, {} test augment'.format(args.model, w_wo))

    loader = StatoilTestLoader(args.test_file, args.batch_size,
                               test_aug=args.augment)()

    n_samples, y_true, ID = get_samples(args)

    if args.cv_path:
        model_files = []
        for f in os.listdir(args.cv_path):
            if f.endswith('.pt'):
                f = os.path.join(args.cv_path, f)
                if os.path.isfile(f):
                    model_files.append(f)
        model_files = sorted(model_files)
        assert len(model_files)
        LOG.info('Found {} models: {}'.format(
            len(model_files), list(map(os.path.basename, model_files))))

        y_pred = np.zeros(n_samples)
        model = get_model(args)
        for model_file in model_files:
            model.load_state_dict(torch.load(model_file))
            y_pred1 = predict_epoch(args, model, loader, n_samples)
            if args.validate:
                show_validate(y_true, y_pred1)
            y_pred += y_pred1
        y_pred /= len(model_files)
    else:
        model = get_model(args)
        model.load_state_dict(torch.load(args.model_file))
        y_pred = predict_epoch(args, model, loader, n_samples)
        if args.validate:
            show_validate(y_true, y_pred)
 
    if not args.validate:
        submit = pd.DataFrame()
        submit['id'] = ID
        submit['is_iceberg'] = y_pred
        submit.to_csv(args.submit_file, index=False, float_format='%.9f')
        print('{} ready to submit!'.format(args.submit_file))


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level = dict(LOGLEVEL)[args.loglevel])

    if not torch.cuda.is_available():
        args.cpu = True
    LOG.debug('Run on {}'.format(['GPU', 'CPU'][args.cpu]))
    args.cuda = not args.cpu

    predict(args)
