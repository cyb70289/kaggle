import os
import subprocess
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


y_names = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']


if __name__ == '__main__':

    def capital_ratio(s):
        small_cnt = len(re.findall(r'[a-z]', s))
        big_cnt = len(re.findall(r'[A-Z]', s))
        alpha_cnt = small_cnt + big_cnt
        if alpha_cnt == 0:
            return 0.0
        return (1.0 * big_cnt) / alpha_cnt

    def gen_X(df):
        # columns: comment_len, bang_cnt, capital_ratio
        X = np.empty((df.shape[0], 3), dtype=np.float32)
        X[:, 0] = df['comment_text'].apply(lambda x: min(len(x.split()), 200))
        X[:, 1] = df['comment_text'].apply(lambda x: min(x.count('!'), 10))
        X[:, 2] = df['comment_text'].apply(capital_ratio)
        return X

    def get_split_indices(num, split_cnt):
        indices = np.arange(num)
        np.random.shuffle(indices)
        indices_lst = []
        if num > split_cnt:
            per_len = num // split_cnt
        else:
            per_len = 1
        s = 0
        for _ in range(split_cnt-1):
            indices_lst.append(indices[s:s+per_len])
            s += per_len
        indices_lst.append(indices[s:])
        return indices_lst

    def do_split(df, split_cnt):
        df_indices = [[] for _ in range(split_cnt)]
        for x in range(1<<len(y_names)):
            dfx = df[df['xxx'] == x]
            indices_lst = get_split_indices(dfx.shape[0], split_cnt)
            for i, indices in enumerate(indices_lst):
                df_indices[i] += dfx.index[indices].tolist()
        return df_indices

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, metavar='N', default=19)
    parser.add_argument('--splits', type=int, metavar='N', default=5)
    args = parser.parse_args()
    np.random.seed(args.seed)

    f = 'dataset/text-embedding.npz'
    if os.path.exists(f):
        print('Text embedding exists: %s' % (f))
    else:
        print('Generating text embedding: %s' % (f))
        subprocess.call(['python',  './text-embedding.py'])

    f = 'dataset/train-split.npz'
    print('Splitting train and validation sets: %s' %(f))
    df = pd.read_csv('dataset/train.csv')
    # one hot to integer
    df['xxx'] = 0
    for col in y_names:
        df['xxx'] *= 2
        df['xxx'] += df[col]
    # split
    indices = do_split(df, args.splits)
    np.savez(f, indices=indices, seed=args.seed)

    sc = MinMaxScaler(copy=False)

    print('Generating train file: dataset/train.npz')
    df = pd.read_csv('dataset/train.csv')
    X = gen_X(df)
    y = df[y_names].values.astype(np.float32)
    sc.fit_transform(X)
    np.savez('dataset/train.npz', X=X, y=y)

    print('Generating test file: dataset/test.npz')
    df = pd.read_csv('dataset/test.csv')
    X = gen_X(df)
    id = df['id']
    sc.transform(X)
    np.savez('dataset/test.npz', X=X, id=id)
