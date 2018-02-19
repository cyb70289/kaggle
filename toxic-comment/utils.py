import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


LOG = logging.getLogger(__name__)

_train_file = 'dataset/train.npz'
_test_file = 'dataset/test.npz'
_embedding_file = 'dataset/text-embedding.npz'
_split_file = 'dataset/train-split.npz'


class ToxicTrainData(Dataset):

    def __init__(self, train_file, train_embedding):
        super(ToxicTrainData, self).__init__()
        self.text = train_embedding
        train_npz = np.load(train_file)
        self.X = train_npz['X'].astype(np.float32)
        self.y = train_npz['y'].astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.text[idx], self.X[idx], self.y[idx]


class ToxicTestData(Dataset):

    def __init__(self, test_file, test_embedding):
        super(ToxicTestData, self).__init__()
        self.text = test_embedding
        test_npz = np.load(test_file)
        self.X = test_npz['X'].astype(np.float32)
        self.id = test_npz['id']

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.text[idx], self.X[idx], 0


class ToxicDataStride(Dataset):

    def __init__(self, dataset, indices, embedding_list):
        super(ToxicDataStride, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.embedding_list = embedding_list
        self.embedding_size = len(embedding_list[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        text, X, y = self.dataset[self.indices[idx]]
        text = self.embedding_list[text]
        return torch.from_numpy(text), torch.from_numpy(X), y


class ToxicTrainSplitter(object):

    def __init__(self, dataset, split_file, embedding_list):
        self.dataset = dataset
        self.embedding_list = embedding_list
        self.indices = np.load(split_file)['indices']

    def split(self):
        train_indices = self.indices[0]
        valid_indices = []
        for indices in self.indices[1:]:
            valid_indices += indices
        train_stride = ToxicDataStride(self.dataset, train_indices,
                                       self.embedding_list)
        valid_stride = ToxicDataStride(self.dataset, valid_indices,
                                       self.embedding_list)
        return train_stride, valid_stride

    def kfold(self):
        for i in range(len(self.indices)):
            train_indices = self.indices[i]
            valid_indices = []
            for j in range(len(self.indices)):
                if j != i:
                    valid_indices += self.indices[j]
            train_stride = ToxicDataStride(self.dataset, train_indices,
                                           self.embedding_list)
            valid_stride = ToxicDataStride(self.dataset, valid_indices,
                                           self.embedding_list)
            yield train_stride, valid_stride


class ToxicTrainLoader(object):

    def __init__(self, batch_size=64, cv=False, n_workers=0):
        self.batch_size = batch_size
        self.cv = cv
        self.n_workers = n_workers

        embedding_npz = np.load(_embedding_file)
        train_embedding = embedding_npz['train_embedding']
        embdding_list = embedding_npz['embedding_list']

        dataset = ToxicTrainData(_train_file, train_embedding)
        self.splitter = ToxicTrainSplitter(dataset, _split_file, embedding_list)

    def __call__(self):
        if self.cv:
            strides = self.splitter.kfold()
            for train_stride, valid_stride in strides:
                train_loader = DataLoader(train_stride, shuffle=True,
                                          batch_size=self.batch_size,
                                          num_workers=self.n_workers)
                valid_loader = DataLoader(valid_stride, shuffle=False,
                                          batch_size=self.batch_size,
                                          num_workers=self.n_workers)
                yield train_loader, valid_loader
        else:
            train_stride, valid_stride = self.splitter.split()
            train_loader = DataLoader(train_stride, shuffle=True,
                                      batch_size=self.batch_size,
                                      num_workers=self.n_workers)
            valid_loader = DataLoader(valid_stride, shuffle=False,
                                      batch_size=self.batch_size,
                                      num_workers=self.n_workers)
            yield train_loader, valid_loader


def ToxicTestLoader(object):

    def __init__(self, batch_size=64, n_workers=0):
        self.batch_size = batch_size
        self.n_workers = n_workers

        embedding_npz = np.load(_embedding_file)
        test_embedding = embedding_npz['test_embedding']
        embdding_list = embedding_npz['embedding_list']

        dataset = ToxicTestData(_test_file, test_embedding)
        indices = np.arange(len(dataset))
        self.stride = ToxicDataStride(dataset, indices, embedding_list)

    def __call__(self):
        return DataLoader(self.stride, batch_size=self.batch_size,
                          num_workers=self.n_workers, shuffle=False)
