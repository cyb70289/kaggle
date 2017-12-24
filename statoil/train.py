import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from models.simplenet import SimpleNet
from models.densenet import DenseNet
from utils import StatoilTrainLoader


_loglevel = (('debug', logging.DEBUG),
             ('info', logging.INFO),
             ('warn', logging.WARN),
             ('error', logging.ERROR))
LOG = logging.getLogger(__name__)

tqdm.monitor_interval = 0


def init_random_seed(seed, cuda):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle statoil competition')
    parser.add_argument('--train-file', default='dataset/train.npz')
    parser.add_argument('--test-file', default='dataset/test.npz')
    parser.add_argument('--model-path', default='saved-models/')
    parser.add_argument('--cpu', action='store_true', help='Train on CPU')
    parser.add_argument('--loglevel', default='info',
                        choices=[x[0] for x in _loglevel])
    parser.add_argument('--seed', type=int, metavar='N', help='Random seed')
    parser.add_argument('--cv', type=int, default=None, help='CV folds')
    parser.add_argument('--batch-size', type=int, metavar='N', default=64)
    parser.add_argument('--max-epochs', type=int, metavar='N', default=666)
    parser.add_argument('--l2', type=float, default=0.0, help='weight decay')
    parser.add_argument('--model', default='simplenet', choices=
                        ['densenet', 'simplenet'])
    return parser.parse_args()


def get_model(args):
    if args.model == 'simplenet':
        model = SimpleNet()
    elif args.model == 'densenet':
        model = DenseNet(block_lst=(4, 4, 4, 4))

    model_path = os.path.join(args.model_path, args.model+'/')
    os.makedirs(model_path, exist_ok=True)

    lossf = nn.BCEWithLogitsLoss(size_average=False)
    optimizer = optim.Adam(model.param_options(), lr=1e-3,
                           weight_decay=args.l2)

    if args.cuda:
        model.cuda()
        lossf.cuda()

    return model, lossf, optimizer, model_path


def train_epoch(args, model, lossf, optimizer, train_loader, dev_loader,
                epoch, max_epochs):
    """ Train and validate one epoch """

    ##################################################
    # Train
    ##################################################
    model.train()

    loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    bar_desc = '{:03d}/{:03d}'.format(epoch+1, max_epochs)
    bar_format = '{l_bar}{bar}| {remaining}{postfix}'
    bar_postfix = '------,---, ------,---'
    pbar = tqdm(total=len(train_loader), desc=bar_desc, bar_format=bar_format)
    pbar.set_postfix_str(bar_postfix)

    for img, label in train_loader:
        model.zero_grad()
        optimizer.zero_grad()

        X = Variable(img, volatile=False, requires_grad=True)
        y = Variable(label, volatile=False, requires_grad=False)
        if args.cuda:
            X = X.cuda(async=True)
            y = y.cuda(async=True)

        # Forward
        predict = model(X)
        output = lossf(predict, y)
        loss_sum += output.data[0]
        acc_sum += ((predict.data > 0.5).float() == y.data).sum()
        count += y.size(0)

        # Backward
        output.backward()
        optimizer.step()

        loss = loss_sum/count
        acc = acc_sum/count*100
        postfix = '{:.4f},{:.0f}%, ------,---'.format(loss, acc)
        pbar.set_postfix_str(postfix)
        pbar.update(1)

    train_loss = loss_sum/count
    train_acc = acc_sum/count

    ##################################################
    # Validate
    ##################################################
    model.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    count = 0

    for img, label in dev_loader:
        X = Variable(img, volatile=True, requires_grad=False)
        y = Variable(label, volatile=True, requires_grad=False)
        if args.cuda:
            X = X.cuda(async=True)
            y = y.cuda(async=True)

        predict = model(X)
        output = lossf(predict, y)
        loss_sum += output.data[0]
        acc_sum += ((predict.data > 0.5).float() == y.data).sum()
        count += y.size(0)

    dev_loss = loss_sum/count
    dev_acc = acc_sum/count
    postfix = '{:.4f},{:.0f}%, {:.4f},{:.0f}%'.format(
        train_loss, train_acc*100, dev_loss, dev_acc*100)
    pbar.set_postfix_str(postfix)

    return train_loss, dev_loss


def train(args):
    LOG.info('Training model: {}'.format(args.model))

    if args.cv:
        loader = StatoilTrainLoader(args.train_file, args.batch_size,
                                    folds=args.cv, seed=args.seed)()
    else:
        loader = StatoilTrainLoader(args.train_file, args.batch_size,
                                    dev_ratio=0.2, seed=args.seed)()

    loss_lst = []

    for i, (train_loader, dev_loader) in enumerate(loader, 1):
        LOG.info('-'*60)
        model, lossf, optimizer, model_path = get_model(args)

        model_file = 'cv{}'.format(i) if args.cv else 'best'
        model_file = os.path.join(model_path, model_file)

        stop_count = 0
        dev_loss_min = 99.99

        for epoch in range(args.max_epochs):
            train_loss, dev_loss = train_epoch(args, model, lossf, optimizer,
                                               train_loader, dev_loader,
                                               epoch, args.max_epochs)
            if dev_loss < dev_loss_min:
                torch.save(model.state_dict(), model_file)
                LOG.info('Model saved: {}, dev_loss: {:.4f}'.format(model_file,
                                                                    dev_loss))
                dev_loss_min = dev_loss
            if train_loss < 0.1:
                # stop training if sure of overfit
                stop_count += 1
                if stop_count >= 3:
                    break

        # append loss to saved file name
        model_file2 = '{}-{:.4f}.pt'.format(model_file, dev_loss_min)
        os.rename(model_file, model_file2)

        loss_lst.append(dev_loss_min)
        del model, lossf

    LOG.info('='*60)
    LOG.info('Loss: {}'.format(list(map(lambda x: '{:.4f}'.format(x),
                                        loss_lst))))
    if args.cv:
        LOG.info('Loss mean: {:.4f}, stdev: {:.4f}'.format(np.mean(loss_lst),
                                                           np.std(loss_lst)))


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level = dict(_loglevel)[args.loglevel])

    if not torch.cuda.is_available():
        args.cpu = True
    LOG.debug('Train on {}'.format(['GPU', 'CPU'][args.cpu]))
    args.cuda = not args.cpu

    init_random_seed(args.seed, args.cuda)

    train(args)
