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

drop_features = [
    'FLAG_DOCUMENT_13', 'NAME_FAMILY_STATUS: Separated',
    'NAME_HOUSING_TYPE: Co-op apartment', 'FLAG_DOCUMENT_14',
    'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
    'FLAG_DOCUMENT_8', 'REG_REGION_NOT_LIVE_REGION',
    'NAME_EDUCATION_TYPE: Incomplete higher',
    'FONDKAPREMONT_MODE: not specified',
    'FONDKAPREMONT_MODE: reg oper spec account',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'FLAG_DOCUMENT_11',
    'WALLSMATERIAL_MODE: Panel', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_5',
    'NAME_INCOME_TYPE: State servant',
    'FONDKAPREMONT_MODE: org spec account', 'FONDKAPREMONT_MODE: na_',
    'FLAG_EMP_PHONE', 'NAME_HOUSING_TYPE: Office apartment',
    'HOUSETYPE_MODE: terraced house', 'WALLSMATERIAL_MODE: Stone, brick',
    'FLAG_DOCUMENT_15', 'WALLSMATERIAL_MODE: Block',
    'EMERGENCYSTATE_MODE: na_', 'HOUSETYPE_MODE: block of flats',
    'WALLSMATERIAL_MODE: Others', 'EMERGENCYSTATE_MODE: No',
    'NAME_HOUSING_TYPE: House / apartment', 'FLAG_CONT_MOBILE',
    'WALLSMATERIAL_MODE: na_', 'EMERGENCYSTATE_MODE: Yes',
    'FONDKAPREMONT_MODE: reg oper account', 'HOUSETYPE_MODE: na_',
    'WALLSMATERIAL_MODE: Monolithic', 'HOUSETYPE_MODE: specific housing',
    'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_10',
    'NAME_EDUCATION_TYPE: Secondary / secondary special',
    'NAME_INCOME_TYPE: Pensioner', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_21',
    'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_19', 'FLAG_MOBIL',
    'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_17',
]


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
    parser.add_argument('--importance', action='store_true')
    parser.add_argument('--finetune', type=int, default=0,
                        help='Random search rounds to finetune parameters')
    return parser.parse_args()


def get_lgb_params():
    return {
        'objective': 'binary',
        'learning_rate': 0.01,
        'seed': lgb_seed,

        'max_bin': 2048,
        'num_leaves': 31,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.5,
        'feature_fraction_seed': lgb_seed+1,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'bagging_seed': lgb_seed+2,
        'lambda_l1': 5,
        'lambda_l2': 2,

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
        'max_bin': ( 1024, 4096, 256 ),
        'num_leaves': (15, 47, 2),
        'min_data_in_leaf': (100, 500, 50),
        'feature_fraction': (0.3, 0.8, 0.05),
        'bagging_fraction': (0.8, 1.0, 0.05),
        'lambda_l1': (1, 8, 1),
        'lambda_l2': (2, 30, 2),
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

    if args.importance:
        features = X_train_val.columns
        importance = np.zeros(len(features))
        for i in range(len(models)):
            _importance = models[i].feature_importance().astype('float32')
            importance += _importance / _importance.sum()
        print('Feature importances:')
        rank = importance.argsort()[::-1]
        for i in rank:
            print('{}:  {:.4f}'.format(features[i], importance[i]))
        print('\n50 least importance features:')
        print(features[rank[-50:]])

    return models


if __name__ == '__main__':
    args = parse_args()
    init_random_seed(args.seed)

    df_train = pickle.load(open(_train_file, 'rb'))
    df_train = df_train.drop(drop_features, axis=1)
    X_train_val = df_train.drop('TARGET', axis=1)
    y_train_val = df_train['TARGET']
    del df_train

    if args.finetune:
        finetune(args, X_train_val, y_train_val)
    else:
        train(args, X_train_val, y_train_val)
