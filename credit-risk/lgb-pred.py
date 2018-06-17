import os
import random
import argparse
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import lgb as lgb_util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=215)
    parser.add_argument('--iter-factor', type=float, default=1.2)
    parser.add_argument('--model-file', help='Model parameter file')
    return parser.parse_args()


def predict1(args, lgb_data_train, X_test, lgb_params, best_iter):
    rounds = 5
    predict = np.zeros(X_test.shape[0])
    for i in range(rounds):
        print('--------------------------------------------------')
        num_boost_rounds = int(round(best_iter * args.iter_factor))
        lgb_params['seed'] += 17
        lgb_params['feature_fraction_seed'] += 17
        lgb_params['bagging_seed'] += 17
        model = lgb.train(lgb_params, lgb_data_train, num_boost_rounds,
                          valid_sets=[lgb_data_train], valid_names=['train'],
                          verbose_eval=lgb_util._verbose_eval)
        predict += model.predict(X_test)
    return predict / rounds


def predict(args, lgb_data_train, X_test):
    predicts = []
    lgb_params = lgb_util.get_lgb_params()

    if args.model_file:
        df = pd.read_csv(args.model_file)
        params = df.columns.drop(['auc', 'stdv', 'best_iter'])
        for i in range(0, df.shape[0]):
            print('==================================================')
            for param in params:
                lgb_params[param] = df.iloc[i, df.columns.get_loc(param)]
            pred = predict1(args, lgb_data_train, X_test, lgb_params,
                            df.iloc[i, df.columns.get_loc('best_iter')])
            predicts.append(pred)
    else:
        pred = predict1(args, lgb_data_train, X_test, lgb_params,
                        lgb_util._best_iter)
        predicts.append(pred)

    return np.array(predicts)


def submit(predicts):
    def ensemble(predicts):
        return predicts.mean(0)

    df_submit = pd.read_csv('dataset/sample_submission.csv')
    df_submit['TARGET'] = ensemble(predicts)
    df_submit.to_csv('dataset/submit_lgb.csv', index=False, float_format='%.9f')
    print('Ready to submit dataset/submit_lgb.csv')


if __name__ == '__main__':
    args = parse_args()
    lgb_util.init_random_seed(args.seed)

    df_train = pickle.load(open(lgb_util._train_file, 'rb'))
    df_train = df_train.drop(lgb_util.drop_features, axis=1)
    X_train = df_train.drop('TARGET', axis=1)
    y_train = df_train['TARGET']
    lgb_data_train = lgb.Dataset(X_train, y_train)
    del df_train, X_train, y_train

    df_test = pickle.load(open(lgb_util._test_file, 'rb'))
    X_test = df_test.drop(lgb_util.drop_features, axis=1)
    del df_test

    predicts = predict(args, lgb_data_train, X_test)
    submit(predicts)
