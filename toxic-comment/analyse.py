import argparse
import numpy as np
import torch
from tqdm import tqdm

from models.simple import SimpleAtt


tqdm.monitor_interval = 0


def load_embedding():
    print('loading word vectors...')
    valid_words = np.load('dataset/valid-words.npz')['valid_words'].item()
    word_lst = []
    embed_lst = []
    with open('dataset/word.vec') as f:
        for row in tqdm(f.readlines()[1:-1]):
            data = row.split(' ')
            word = data[0]
            if word in valid_words:
                embed = np.array([float(num) for num in data[1:-1]],
                                 dtype=np.float32)
                word_lst.append(word)
                embed_lst.append(embed)

    return word_lst, np.stack(embed_lst, axis=0)


def parse_simple_atten(model_file, **kwargs):
    ''' show words match simple atten most '''

    embed_dim = kwargs.get('embed_dim', 300)
    text_len = kwargs.get('text_len', 128)
    cuda = kwargs.get('cuda', False)

    model = SimpleAtt(embed_dim, text_len, 6, cuda=cuda)
    model.load_state_dict(torch.load(model_file))
    for name, param in model.named_parameters():
        if name == 'query_weights':
            weights = param.data.numpy()
            break

    words, embeddings = load_embedding()
    embeddings *= weights
    embeddings = embeddings.sum(axis=1)
    idx = embeddings.argsort()
    for i in idx[-1:-201:-1]:
        print(words[i])


parser = argparse.ArgumentParser()
parser.add_argument('--model-file')
args = parser.parse_args()

cuda = torch.cuda.is_available()
parse_simple_atten(args.model_file, cuda=cuda)
