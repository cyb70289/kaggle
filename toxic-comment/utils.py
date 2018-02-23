import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


LOGLEVEL = (('debug', logging.DEBUG),
            ('info', logging.INFO),
            ('warn', logging.WARN),
            ('error', logging.ERROR))

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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.text[idx], self.X[idx], 0


class ToxicDataStride(Dataset):

    def __init__(self, dataset, indices, embedding_list):
        super(ToxicDataStride, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.embedding_list = embedding_list

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
        valid_indices = self.indices[0]
        train_indices = []
        for indices in self.indices[1:]:
            train_indices += indices
        train_stride = ToxicDataStride(self.dataset, train_indices,
                                       self.embedding_list)
        valid_stride = ToxicDataStride(self.dataset, valid_indices,
                                       self.embedding_list)
        return train_stride, valid_stride

    def kfold(self):
        for i in range(len(self.indices)):
            valid_indices = self.indices[i]
            train_indices = []
            for j in range(len(self.indices)):
                if j != i:
                    train_indices += self.indices[j]
            train_stride = ToxicDataStride(self.dataset, train_indices,
                                           self.embedding_list)
            valid_stride = ToxicDataStride(self.dataset, valid_indices,
                                           self.embedding_list)
            yield train_stride, valid_stride


class ToxicTrainLoader(object):

    def __init__(self, batch_size, cv, n_workers=0):
        self.batch_size = batch_size
        self.cv = cv
        self.n_workers = n_workers

        embedding_npz = np.load(_embedding_file)
        train_embedding = embedding_npz['train_embedding']
        embedding_list = embedding_npz['embedding_list']

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


class ToxicTestLoader(object):

    def __init__(self, batch_size, n_workers=0, validate=False):
        self.batch_size = batch_size
        self.n_workers = n_workers

        embedding_npz = np.load(_embedding_file)
        train_embedding = embedding_npz['train_embedding']
        test_embedding = embedding_npz['test_embedding']
        embedding_list = embedding_npz['embedding_list']

        if validate:
            dataset = ToxicTestData(_train_file, train_embedding)
        else:
            dataset = ToxicTestData(_test_file, test_embedding)
        indices = np.arange(len(dataset))
        self.stride = ToxicDataStride(dataset, indices, embedding_list)

    def __call__(self):
        return DataLoader(self.stride, batch_size=self.batch_size,
                          num_workers=self.n_workers, shuffle=False)


############################################################
# learning rate scheduler
############################################################

class LRSchedNone(object):
    """ No learning rate adjustment """

    def __init__(self, param_groups, lr):
        self.param_groups = param_groups
        self.set_lr(lr)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def update(self, loss, **kwargs):
        pass


class LRSchedStep(LRSchedNone):
    """ Learning rate scheduler based on predefine (loss, lr) pairs """

    def __init__(self, param_groups, lr, *steps):
        super(LRSchedStep, self).__init__(param_groups, lr)
        self.steps = steps

    def update(self, loss, **kwargs):
        maximize = kwargs.get('maximize', False)
        for step_loss, step_lr in self.steps:
            adjust_needed = (maximize and loss > step_loss) or \
                ((not maximize) and loss < step_loss)
            if adjust_needed and self.lr > step_lr:
                self.set_lr(step_lr)
                LOG.info('Update learning rate to {:.5f}'.format(step_lr))


class LRSchedDecay(LRSchedNone):
    """ Learning rate decay on each epoch """

    def __init__(self, param_groups, lr, decay, lr_min=0.0):
        super(LRSchedDecay, self).__init__(param_groups, lr)
        self.decay = decay
        self.lr_min = lr_min

    def update(self, loss, **kwargs):
        if self.lr > self.lr_min:
            lr = max(self.lr*self.decay, self.lr_min)
            self.set_lr(lr)
            LOG.debug('Update learning rate to {:.5f}'.format(lr))
