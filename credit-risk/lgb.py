import os
import random
import argparse
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    parser.add_argument('--finetune', type=int, default=0,
                        help='Random search rounds to finetune parameters')
    return parser.parse_args()


def get_lgb_params():
    return {
        'objective': 'binary',
        'learning_rate': 0.01,
        'seed': lgb_seed,

        'max_depth': 5,
        'num_leaves': 32,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.5,
        'feature_fraction_seed': lgb_seed+1,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'bagging_seed': lgb_seed+2,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,

        'verbosity': -1,
        'metric': ['auc', 'binary_logloss'],
    }


def _train(args, lgb_params, X_train_val, y_train_val):
    models = []
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
        models.append(model)

    logloss = log_loss(y_train_val, y_predict)
    auc = roc_auc_score(y_train_val, y_predict)
    print('\n==========================================================')
    print('logloss: {:.4f}, auc: {:.4f}'.format(logloss, auc))

    return models, (logloss, auc)


def finetune(args, X_train_val, y_train_val):
    def tune(lgb_params, random_params, params, randsteps_lst):
        while True:
            randsteps = []
            for k in params:
                v = random_params[k]
                gap, step = v[1]-v[0], v[2]
                randstep = random.randint(0, round(gap/step))
                lgb_params[k] = v[0] + randstep*step
                randsteps.append(randstep)
            for _randsteps in randsteps_lst:
                if _randsteps == randsteps:
                    break
            else:
                randsteps_lst.append(randsteps)
                break
        s = ''
        for k in params:
            s = s + k + ':'
            v = lgb_params[k]
            s += '{}'.format(v) if type(v) == int else '{:.2f}'.format(v)
            s += ', '
        print(s)

    if os.path.exists('dataset/finetune.csv'):
        os.rename('dataset/finetune.csv', 'dataset/finetune-old.csv')

    random.seed(None)

    best_logloss = 999.0
    best_auc = 0.0
    randsteps_lst = []

    lgb_params = get_lgb_params()
    random_params = {
        'max_depth': (4, 5, 1),
        'num_leaves': (16, 32, 1),
        'min_data_in_leaf': (20, 200, 10),
        'feature_fraction': (0.4, 0.8, 0.05),
        'bagging_fraction': (0.6, 1.0, 0.05),
        'lambda_l1': (0.5, 2.0, 0.1),
        'lambda_l2': (0.5, 2.0, 0.1),
    }

    with open('dataset/finetune.csv', 'w', 1) as f:
        params = sorted(list(random_params.keys()))
        f.write(','.join(params))
        f.write(',logloss,auc\n')
        for i in range(args.finetune):
            print('Iteration {}/{}'.format(i+1, args.finetune))
            tune(lgb_params, random_params, params, randsteps_lst)
            _, (logloss, auc) = _train(args, lgb_params,
                                       X_train_val, y_train_val)
            for param in params:
                v = lgb_params[param]
                if type(v) == int:
                    f.write('{},'.format(v))
                else:
                    f.write('{:.2f},'.format(v))
            f.write('{:.4f},{:.4f}\n'.format(logloss, auc))
            os.fsync(f.fileno())
            if logloss < best_logloss:
                best_logloss = logloss
            if best_auc < auc:
                best_auc = auc
            print('Best logloss: {:.4f}, best auc: {:.4f}'.format(
                best_logloss, best_auc))


def train(args, X_train_val, y_train_val):
    models, _ = _train(args, get_lgb_params(), X_train_val, y_train_val)
    return models


if __name__ == '__main__':
    args = parse_args()
    init_random_seed(args.seed)

    df_train = pickle.load(open(_train_file, 'rb'))
    X_train_val = df_train.drop('TARGET', axis=1)
    y_train_val = df_train['TARGET']
    del df_train

    if args.finetune:
        finetune(args, X_train_val, y_train_val)
    else:
        train(args, X_train_val, y_train_val)
