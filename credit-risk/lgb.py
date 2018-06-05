import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import utils
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


_train_file = 'dataset/df_train.pkl'
_test_file = 'dataset/df_test.pkl'

lgb_seed = 0
sklearn_seed = None


def init_random_seed(seed):
    global lgb_seed, sklearn_seed
    if seed:
        np.random.seed(seed)
        sklearn_seed = seed+100
        lgb_seed = seed+200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=9999)
    parser.add_argument('--stop-count', type=int, default=333,
                        help='Early stop count')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Cross validation folds')
    parser.add_argument('--seed', type=int, default=1124)
    parser.add_argument('--loglevel', default='info',
                        choices=[x[0] for x in utils.LOGLEVEL])
    return parser.parse_args()


def train(args):
    df_train = pickle.load(open(_train_file, 'rb'))
    X_train_val = df_train.drop('TARGET', axis=1)
    y_train_val = df_train['TARGET']
    del df_train

    lgb_params = {
        'objective': 'binary',
        'learning_rate': 0.01,
        'seed': lgb_seed,
        'max_depth': 5,
        'num_leaves': 32,
        'feature_fraction': 0.8,
        'feature_fraction_seed': lgb_seed+1,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'bagging_seed': lgb_seed+2,
        'metric': ['auc', 'binary_logloss'],
        'verbosity': -1,
    }

    best_iter = 0
    y_predict = np.zeros(X_train_val.shape[0])
    kf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                         random_state=sklearn_seed)

    for train_index, val_index in kf.split(X_train_val, y_train_val):
        X_train, X_val = \
            X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :]
        y_train, y_val = \
            y_train_val.values[train_index], y_train_val.values[val_index]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        print('---------------------------------------------------------')
        model = lgb.train(lgb_params, lgb_train,
                          num_boost_round=args.max_epochs,
                          valid_sets=[lgb_train, lgb_val],
                          valid_names=['train', 'val'],
                          early_stopping_rounds=args.stop_count,
                          verbose_eval=200)
        y_predict[val_index] = model.predict(
            X_val, num_iteration=model.best_iteration)
        best_iter += model.best_iteration

    print('\n==========================================================')
    print('mean best iter: {}'.format(best_iter//args.cv_folds))
    print('logloss: {:.4f}, auc: {:.4f}'.format(
        log_loss(y_train_val, y_predict),
        roc_auc_score(y_train_val, y_predict)))


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level = dict(utils.LOGLEVEL)[args.loglevel])
    init_random_seed(args.seed)
    train(args)
