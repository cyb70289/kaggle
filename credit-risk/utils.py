import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


LOGLEVEL = (('debug', logging.DEBUG),
            ('info', logging.INFO),
            ('warn', logging.WARN),
            ('error', logging.ERROR))

LOG = logging.getLogger(__name__)

_train_file = 'dataset/df_train_norm.pkl'
_test_file = 'dataset/df_test_norm.pkl'


class CreditTrainData(Dataset):

    def __init__(self, train_file):
        super(CreditTrainData, self).__init__()
        data = pickle.load(open(train_file, 'rb')).values
        # last column is TARGET
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CreditTestData(Dataset):

    def __init__(self, test_file):
        super(CreditTestData, self).__init__()
        self.X = pickle.load(open(test_file, 'rb')).values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], 0


class CreditDataStride(Dataset):

    def __init__(self, dataset, indices):
        super(CreditDataStride, self).__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        X, y = self.dataset[self.indices[idx]]
        return torch.from_numpy(X), y


class CreditTrainSplitter(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.y = dataset.y.squeeze()
        self.X = np.zeros(len(self.y))

    def split(self, valid_ratio, stratified=True, seed=None):
        CVClass = StratifiedShuffleSplit if stratified else ShuffleSplit
        cv = CVClass(test_size=valid_ratio, random_state=seed)
        train_index, valid_index = next(cv.split(self.X, self.y))
        train_stride = CreditDataStride(self.dataset, train_index)
        valid_stride = CreditDataStride(self.dataset, valid_index)
        return train_stride, valid_stride

    def kfold(self, folds, stratified=True, seed=None):
        KFClass = StratifiedKFold if stratified else KFold
        kf = KFClass(n_splits=folds, shuffle=True, random_state=seed)
        for train_index, valid_index in kf.split(self.X, self.y):
            train_stride = CreditDataStride(self.dataset, train_index)
            valid_stride = CreditDataStride(self.dataset, valid_index)
            yield train_stride, valid_stride


class CreditTrainLoader(object):

    def __init__(self, batch_size, valid_ratio=None, folds=None,
                 seed=None, n_workers=0):
        if valid_ratio is None and folds is None:
            raise ValueError('Either "valid_ratio" or "folds" must be set')
        elif valid_ratio and folds:
            raise ValueError('Cannot set both "valid_ratio" and "folds"')
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.folds = folds
        self.seed = seed
        self.n_workers = n_workers

        dataset = CreditTrainData(_train_file)
        self.splitter = CreditTrainSplitter(dataset)

    def __call__(self):
        if self.folds:
            strides = self.splitter.kfold(folds=self.folds, seed=self.seed)
            for train_stride, valid_stride in strides:
                train_loader = DataLoader(train_stride, shuffle=True,
                                          batch_size=self.batch_size,
                                          num_workers=self.n_workers)
                valid_loader = DataLoader(valid_stride, shuffle=False,
                                          batch_size=self.batch_size,
                                          num_workers=self.n_workers)
                yield train_loader, valid_loader
        else:
            train_stride, valid_stride = self.splitter.split(
                valid_ratio=self.valid_ratio, seed=self.seed)
            train_loader = DataLoader(train_stride, shuffle=True,
                                      batch_size=self.batch_size,
                                      num_workers=self.n_workers)
            valid_loader = DataLoader(valid_stride, shuffle=False,
                                      batch_size=self.batch_size,
                                      num_workers=self.n_workers)
            yield train_loader, valid_loader


class CreditTestLoader(object):

    def __init__(self, batch_size, n_workers=0):
        self.batch_size = batch_size
        self.n_workers = n_workers

        dataset = CreditTestData(_test_file)
        indices = np.arange(len(dataset))
        self.stride = CreditDataStride(dataset, indices)

    def __call__(self):
        return DataLoader(self.stride, batch_size=self.batch_size,
                          num_workers=self.n_workers, shuffle=False)
