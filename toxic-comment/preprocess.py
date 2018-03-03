import os
import subprocess
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


y_names = ['toxic', 'severe_toxic', 'obscene',
           'threat', 'insult', 'identity_hate']

upsample_id = 'xxxxxxxxxxxxxxxx'


def upsample():
    ''' upsample rare classes
        - severe_toxic:  x10
        - obscene:       x2
        - threat:        x30
        - insult:        x2
        - identity_hate: x10
    '''

    def gen_comment(comments):
        new_comment = ''
        # 10 to 20 sentences
        n_sentences = np.random.randint(5, 11)
        for _ in range(n_sentences):
            comment = np.random.choice(comments, 1)[0]
            sentences = re.split('[.!?;\n]', comment)
            if len(sentences) == 0:
                continue
            # pick 2 consecutive sentences from first 10
            idx = np.random.randint(0, min(10, len(sentences)))
            new_comment += sentences[idx]
            if (idx+1) < len(sentences):
                new_comment += sentences[idx+1]
                new_comment += '. '
        return new_comment

    def upsample1(df, col, ratio):
        print('upsamping', col)
        cnt = df[col].sum() * (ratio-1)
        df_new = pd.DataFrame(index=np.arange(cnt), columns=df.columns)
        for i in range(cnt):
            comments = df[df[col]==1]['comment_text'].values
            comment = gen_comment(comments)
            df_new.loc[i] = [upsample_id, comment, 0, 0, 0, 0, 0, 0]
        df_new[col] = 1
        return df.append(df_new, ignore_index=True, verify_integrity=True)

    df = pd.read_csv('dataset/train.csv')
    df = upsample1(df, 'severe_toxic', 10)
    df = upsample1(df, 'obscene', 2)
    df = upsample1(df, 'threat', 30)
    df = upsample1(df, 'insult', 2)
    df = upsample1(df, 'identity_hate', 10)
    df.to_csv('dataset/train-upsample.csv', index=False)


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, metavar='N', default=19)
    parser.add_argument('--splits', type=int, metavar='N', default=5)
    parser.add_argument('--upsample', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    np.random.seed(args.seed)

    train_csv = 'dataset/train.csv'
    if args.upsample:
        train_csv = 'dataset/train-upsample.csv'
        if os.path.exists(train_csv):
            print('Upsampled dataset exists: %s' % (train_csv))
        else:
            print('Upsampling train dataset...')
            upsample()

    f = 'dataset/text-embedding.npz'
    if not args.force and os.path.exists(f):
        print('Text embedding exists: %s' % (f))
    else:
        print('Generating text embedding: %s' % (f))
        rc = subprocess.call(['python', './text-embedding.py', train_csv])
        assert(rc == 0)

    f = 'dataset/train-split.npz'
    print('Splitting train and validation sets: %s' % (f))
    df = pd.read_csv(train_csv)
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
    df = pd.read_csv(train_csv)
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
