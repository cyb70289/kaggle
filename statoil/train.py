import argparse
import logging
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from models.densenet import DenseNet
from utils import StatoilTrainLoader


_loglevel = (('debug', logging.DEBUG),
             ('info', logging.INFO),
             ('warn', logging.WARN),
             ('error', logging.ERROR))

LOG = logging.getLogger(__name__)


def init_random_seed(seed):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle statoil competition')
    parser.add_argument('--train-file', default='dataset/train.npz')
    parser.add_argument('--test-file', default='dataset/test.npz')
    parser.add_argument('--cpu', action='store_true', help='Train on CPU')
    parser.add_argument('--loglevel', default='info',
                        choices=[x[0] for x in _loglevel])
    parser.add_argument('--seed', type=int, metavar='N', help='Random seed')
    parser.add_argument('--cv', type=int, default=None, help='CV folds')
    parser.add_argument('--batch-size', type=int, metavar='N', default=64)
    parser.add_argument('--max-epochs', type=int, metavar='N', default=100)
    parser.add_argument('--l2', type=float, default=0.0, help='weight decay')
    return parser.parse_args()


def train_epoch(args, model, loader, loss, optimizer):
    model.train()

    bce_sum = 0.0
    bce_count = 0

    LOG.info('Training...')

    for img, label in loader:
        model.zero_grad()
        optimizer.zero_grad()

        X = Variable(img, volatile=False, requires_grad=True)
        y = Variable(label, volatile=False, requires_grad=False)
        if args.cuda:
            X = X.cuda(async=True)
            y = y.cuda(async=True)
        # Forward
        output = loss(model(X), y)
        bce_sum += output.data[0]
        bce_count += y.size(0)
        # Backward
        output.backward()
        optimizer.step()

    LOG.info('Train loss: {:.4f}'.format(bce_sum/bce_count))


def validate_epoch(args, model, loader, loss, optimizer):
    model.eval()

    bce_sum = 0.0
    bce_count = 0

    LOG.info('Validating...')

    for img, label in loader:
        X = Variable(img, volatile=True, requires_grad=False)
        y = Variable(label, volatile=True, requires_grad=False)
        if args.cuda:
            X = X.cuda(async=True)
            y = y.cuda(async=True)
        # Evaluate
        output = loss(model(X), y)
        bce_sum += output.data[0]
        bce_count += y.size(0)

    LOG.info('Dev loss: {:.4f}'.format(bce_sum/bce_count))


def train(args, model):
    loss = nn.BCEWithLogitsLoss(size_average=False)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                          nesterov=True, weight_decay=args.l2)

    if args.cuda:
        model.cuda()
        loss.cuda()

    Loader = StatoilTrainLoader(args.train_file, args.batch_size,
                                dev_ratio=0.2, seed=args.seed)
    train_loader, dev_loader = next(Loader())
    for i in range(args.max_epochs):
        bceloss_train = train_epoch(args, model, train_loader, loss, optimizer)
        bceloss_dev = validate_epoch(args, model, dev_loader, loss, optimizer)


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level = dict(_loglevel)[args.loglevel])

    if not torch.cuda.is_available():
        args.cpu = True
    LOG.debug('Train on {}'.format(['GPU', 'CPU'][args.cpu]))
    args.cuda = not args.cpu

    init_random_seed(args.seed)

    model = DenseNet()
    if args.cv:
        train_cv(args, model)
    else:
        train(args, model)
