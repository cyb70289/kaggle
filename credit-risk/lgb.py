import os
import random
import argparse
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


_train_file = 'dataset/df_train.pkl'
_test_file = 'dataset/df_test.pkl'
_verbose_eval = 100
_lgb_seed = 0
_sklearn_seed = None

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
    global _lgb_seed
    if seed:
        np.random.seed(seed)
        _lgb_seed = seed+100
        _sklearn_seed = seed+200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=9999)
    parser.add_argument('--early-stop-rounds', type=int, default=333)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1124)
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate default parameters')
    parser.add_argument('--tune', type=int, default=0,
                        help='Random search rounds to finetune parameters')
    parser.add_argument('--importance', action='store_true',
                        help='Show feature importances')
    return parser.parse_args()


def get_lgb_params():
    return {
        'objective': 'binary',
        'learning_rate': 0.01,
        'seed': _lgb_seed,

        'max_bin': 2048,
        'num_leaves': 31,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'lambda_l1': 5,
        'lambda_l2': 2,

        'verbosity': -1,
        'metric': 'auc',
    }


def train(args, X_train_val, y_train_val):
    y_predict = np.zeros(X_train_val.shape[0])
    kf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                         random_state=_sklearn_seed)
    if args.importance:
        features = X_train_val.columns
        importance = np.zeros(len(features))

    for train_index, val_index in kf.split(X_train_val, y_train_val):
        X_train, X_val = \
            X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :]
        y_train, y_val = \
            y_train_val.values[train_index], y_train_val.values[val_index]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        print('---------------------------------------------------------')
        model = lgb.train(get_lgb_params(), lgb_train, args.max_epochs,
                          valid_sets=[lgb_train, lgb_val],
                          valid_names=['train', 'val'],
                          early_stopping_rounds=args.early_stop_rounds,
                          verbose_eval=_verbose_eval)
        y_predict[val_index] = model.predict(
            X_val, num_iteration=model.best_iteration)
        if args.importance:
            _importance = model.feature_importance().astype('float32')
            importance += _importance / _importance.sum()

    print('\n==========================================================')
    print('auc: {:.4f}'.format(roc_auc_score(y_train_val, y_predict)))

    if args.importance:
        print('\nFeature importances:')
        rank = importance.argsort()[::-1]
        for i in rank:
            print('{}:  {:.4f}'.format(features[i], importance[i]))
        print('\n50 least importance features:')
        print(features[rank[-50:]])


def _evaluate(args, lgb_params, lgb_data):
    hist = lgb.cv(lgb_params, lgb_data, num_boost_round=args.max_epochs,
                  nfold=args.cv_folds, stratified=True, shuffle=True,
                  early_stopping_rounds=args.early_stop_rounds,
                  seed=_lgb_seed+1, verbose_eval=_verbose_eval)
    max_auc = max(hist['auc-mean'])
    best_iter = hist['auc-mean'].index(max_auc)+1
    stdv = hist['auc-stdv'][best_iter-1]
    print('--------------------------------------------------')
    print('auc: {:.4f}, std: {:.4f}, best_iter: {}'.format(max_auc, stdv,
                                                           best_iter))
    return max_auc, stdv, best_iter


def finetune(args, lgb_data):
    def tune(lgb_params, random_params, param_names):
        s = ''
        for k in param_names:
            startv, endv, step = random_params[k]
            randstep = random.randint(0, round((endv-startv)/step))
            lgb_params[k] = v = startv + randstep*step
            s = s + k + ':'
            s += '{}'.format(v) if type(v) == int else '{:.2f}'.format(v)
            s += ', '
        print(s)

    if os.path.exists('dataset/finetune.csv'):
        os.rename('dataset/finetune.csv', 'dataset/finetune-old.csv')

    random.seed(None)

    lgb_params = get_lgb_params()
    random_params = {
        'max_bin': ( 1024, 4096, 256 ),
        'num_leaves': (16, 32, 2),
        'min_data_in_leaf': (100, 500, 50),
        'feature_fraction': (0.3, 0.8, 0.05),
        'bagging_fraction': (0.8, 1.0, 0.05),
        'lambda_l1': (1, 8, 1),
        'lambda_l2': (2, 30, 4),
    }

    with open('dataset/finetune.csv', 'w', 1) as f:
        param_names = sorted(list(random_params.keys()))
        f.write(','.join(param_names))
        f.write(',auc,stdv,best_iter\n')
        for i in range(args.tune):
            print('\n==================================================')
            print('Iteration {}/{}'.format(i+1, args.tune))
            tune(lgb_params, random_params, param_names)
            auc, stdv, best_iter = _evaluate(args, lgb_params, lgb_data)
            for param in param_names:
                v = lgb_params[param]
                if type(v) == int:
                    f.write('{},'.format(v))
                else:
                    f.write('{:.2f},'.format(v))
            f.write('{:.4f},{:.4f},{}\n'.format(auc, stdv, best_iter))
            os.fsync(f.fileno())


def evaluate(args, lgb_data):
    _evaluate(args, get_lgb_params(), lgb_data)


if __name__ == '__main__':
    args = parse_args()
    init_random_seed(args.seed)

    df_train = pickle.load(open(_train_file, 'rb'))
    df_train = df_train.drop(drop_features, axis=1)
    X_train_val = df_train.drop('TARGET', axis=1)
    y_train_val = df_train['TARGET']
    lgb_data = lgb.Dataset(X_train_val, y_train_val)
    del df_train

    if args.importance:
        args.tune = 0
        args.eval = False

    if args.tune > 0:
        finetune(args, lgb_data)
    elif args.eval:
        evaluate(args, lgb_data)
    else:
        train(args, X_train_val, y_train_val)
