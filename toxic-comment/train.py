import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from torch.autograd import Variable

from models.rnn_att import RnnAtt
import utils
from utils import ToxicTrainLoader, LRSchedNone
import preprocess


_model_path = './saved-models/'
_n_classes = len(preprocess.y_names)

LOG = logging.getLogger(__name__)

tqdm.monitor_interval = 0


def init_random_seed(seed, cuda):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--stop-count', type=int, default=3,
                        help='Early stop count')
    parser.add_argument('--l2', type=float, default=0.0, help='weight decay')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--rnn-model', default='gru', choices= ['gru', 'lstm'])
    parser.add_argument('--rnn-hidden-dim', type=int, default=512)
    parser.add_argument('--rnn-attention', action='store_true')
    parser.add_argument('--text-len', type=int, default=128)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--seed', type=int, metavar='N', help='Random seed')
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


def get_model(args):
    model = RnnAtt(args.embed_dim, args.rnn_hidden_dim, args.text_len,
                   _n_classes, model=args.rnn_model, bidir=args.bidirectional,
                   atten=args.rnn_attention, cuda=args.cuda)

    model_path = os.path.join(_model_path, args.rnn_model+'/')
    os.makedirs(model_path, exist_ok=True)

    lossf = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.param_options(), weight_decay=args.l2)
    lrsched = LRSchedNone(optimizer.param_groups, 1e-3)

    if args.cuda:
        model.cuda()
        lossf.cuda()

    return model, lossf, optimizer, lrsched, model_path


def train_epoch(args, model, lossf, optimizer, train_loader, valid_loader,
                epoch, max_epochs):
    ##################################################
    # Train
    ##################################################
    model.train()

    n_samples = len(train_loader.dataset)
    y_true = np.zeros((n_samples, _n_classes), dtype=np.float32)
    y_score = np.zeros_like(y_true)
    y_cnt = 0
    loss_sum = 0.0

    bar_desc = '{:d}/{:d}'.format(epoch+1, max_epochs)
    bar_format = '{l_bar}{bar}| {remaining}{postfix}'
    bar_postfix = '------,---, ------,---'
    pbar = tqdm(total=len(train_loader), desc=bar_desc, bar_format=bar_format)
    pbar.set_postfix_str(bar_postfix)

    for text, X, y in train_loader:
        model.zero_grad()
        optimizer.zero_grad()

        y_true[y_cnt:y_cnt+y.size(0)] = y.numpy()

        text = text[:, :args.text_len, :]
        text = Variable(text, requires_grad=False)
        X = Variable(X, requires_grad=False)
        y = Variable(y, requires_grad=False)
        if args.cuda:
            text = text.contiguous().cuda(async=True)
            X = X.cuda(async=True)
            y = y.cuda(async=True)

        # Forward
        predict = model(text, X)
        output = lossf(predict, y)

        # Backward
        output.backward()
        optimizer.step()

        y_score[y_cnt:y_cnt+y.size(0)] = predict.data.cpu().numpy()
        y_cnt += y.size(0)

        loss_sum += output.data[0] * y.size(0)
        loss = loss_sum / y_cnt
        try:
            auc = roc_auc_score(y_true[:y_cnt], y_score[:y_cnt])
        except ValueError:
            auc = 0.0

        postfix = '{:.4f},{:.4f}, ------,------'.format(loss, auc)
        pbar.set_postfix_str(postfix)
        pbar.update(1)

    assert(y_cnt == n_samples)
    train_loss = loss_sum / y_cnt
    train_auc = roc_auc_score(y_true, y_score)

    ##################################################
    # Validate
    ##################################################
    model.eval()

    n_samples = len(valid_loader.dataset)
    y_true = np.zeros((n_samples, _n_classes), dtype=np.float32)
    y_score = np.zeros_like(y_true)
    y_cnt = 0
    loss_sum = 0.0

    for text, X, y in valid_loader:
        y_true[y_cnt:y_cnt+y.size(0)] = y.numpy()

        text = text[:, :args.text_len, :]
        text = Variable(text, requires_grad=False)
        X = Variable(X, requires_grad=False)
        y = Variable(y, requires_grad=False)
        if args.cuda:
            text = text.contiguous().cuda(async=True)
            X = X.cuda(async=True)
            y = y.cuda(async=True)

        predict = model(text, X)
        output = lossf(predict, y)

        loss_sum += output.data[0] * y.size(0)
        y_score[y_cnt:y_cnt+y.size(0)] = predict.data.cpu().numpy()
        y_cnt += y.size(0)

    assert(y_cnt == n_samples)
    valid_loss = loss_sum / y_cnt
    valid_auc = roc_auc_score(y_true, y_score)

    postfix = '{:.4f},{:.4f}, {:.4f},{:.4f}'.format(
        train_loss, train_auc, valid_loss, valid_auc)
    pbar.set_postfix_str(postfix)

    return valid_auc


def train(args):
    LOG.info('Training model: {}'.format(args.rnn_model))

    auc_lst = []
    loader = ToxicTrainLoader(args.batch_size, args.cv, 4)()

    for i, (train_loader, valid_loader) in enumerate(loader, 1):
        LOG.info('-'*60)
        model, lossf, optimizer, lrsched, model_path = get_model(args)

        model_file = 'cv{}'.format(i) if args.cv else 'best'
        model_file = os.path.join(model_path, model_file)

        no_improve_count = 0
        valid_auc_min = 0.0

        for epoch in range(args.max_epochs):
            valid_auc = train_epoch(args, model, lossf, optimizer,
                                    train_loader, valid_loader,
                                    epoch, args.max_epochs)
            if valid_auc > valid_auc_min:
                torch.save(model.state_dict(), model_file)
                LOG.info('Model saved: {}, valid_auc: {:.4f}'.format(
                    model_file, valid_auc))
                valid_auc_min = valid_auc
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= args.stop_count:
                LOG.info('Early stopping')
                break

            lrsched.update(valid_auc)

        model_file2 = '{}-{}-{:.4f}.pt'.format(model_file, args.text_len,
                                               valid_auc_min)
        os.rename(model_file, model_file2)

        auc_lst.append(valid_auc_min)

    LOG.info('='*60)
    LOG.info('AUC: {}'.format(list(map(lambda x: '{:.4f}'.format(x), auc_lst))))
    if args.cv:
        LOG.info('AUC mean: {:.4f}, stdev: {:.4f}'.format(np.mean(auc_lst),
                                                          np.std(auc_lst)))


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level = dict(utils.LOGLEVEL)[args.loglevel])

    args.cuda = torch.cuda.is_available()
    LOG.debug('Train on {}'.format('GPU' if args.cuda else 'CPU'))

    init_random_seed(args.seed, args.cuda)
    get_embedding_info(args)
    train(args)
