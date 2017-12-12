import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold


LOG = logging.getLogger(__name__)


class _SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class _TransformAugment(object):

    def __init__(self):
        self.aug_funcs = [getattr(self, f) for f in _TransformAugment.__dict__
                          if f.startswith('_aug_')]
        self.aug_funcs = [f for f in self.aug_funcs if callable(f)]
        LOG.debug('Image augment: {}'.format(
            [f.__name__ for f in self.aug_funcs]))

    def _aug_flip(self, img):
        # 0 - no, 1 - vertical, 2 - horizontal, 3 - both
        i = np.random.randint(0, 4)
        if i & 1:
            img = np.flip(img, 0)
        if i & 2:
            img = np.flip(img, 1)
        return img

    def _aug_rot90(self, img):
        assert(img.shape[0] == img.shape[1])
        # 0 - 0, 1 - 90, 2 - 180, 3 - 270
        i = np.random.randint(0, 4)
        return np.rot90(img, i)

    def __call__(self, img):
        # img shape: H * W * C
        for aug_func in self.aug_funcs:
            img = aug_func(img)
        return img


class _TransformToTensor(object):

    def __call__(self, img):
        # H * W * C --> C * H * W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img.copy())


class _TrainDevSplitter(object):

    def __init__(self, labels, seed=None):
        self.y = labels.squeeze()
        self.X = np.zeros(len(self.y))
        self.seed = seed

    def split(self, dev_ratio=0.2, stratified=True):
        CVClass = StratifiedShuffleSplit if stratified else ShuffleSplit
        cv = CVClass(test_size=dev_ratio, random_state=self.seed)
        return next(cv.split(self.X, self.y))

    def kfold(self, folds=5, stratified=True):
        KFClass = StratifiedKFold if stratified else KFold
        kf = KFClass(n_splits=folds, shuffle=True, random_state=self.seed)
        train_indices, dev_indices = [], []
        for train_index, dev_index in kf.split(self.X, self.y):
            train_indices.append(train_index)
            dev_indices.append(dev_index)
        return train_indices, dev_indices


class _TrainDevData(Dataset):

    def __init__(self, train_npz, seed=None, aug=True):
        data = np.load(train_npz)
        self.imgs = data['img'].astype(np.float32)
        self.labels = data['y_train'].astype(np.float32)[..., None]
        self.splitter = _TrainDevSplitter(self.labels, seed)
        self.transform = _TransformToTensor()
        if aug:
            self.transform = transforms.Compose([_TransformAugment(),
                                                 self.transform])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])
        label = self.labels[idx]
        return (img, label)


class _TestData(Dataset):

    def __init__(self, test_npz, aug=False):
        data = np.load(test_npz)
        self.imgs = data['img'].astype(np.float32)
        self.transform = _TransformToTensor()
        if aug:
            self.transform = transforms.Compose([_TransformAugment(),
                                                 self.transform])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx])


class TrainDevLoader(object):

    def __init__(self, train_npz, batch_size=1, dev_ratio=None, folds=None,
                 stratified=True, seed=None, n_workers=0, aug=True):
        if dev_ratio is None and folds is None:
            raise ValueError('Either "dev_ratio" or "folds" must be set!')
        elif dev_ratio and folds:
            raise ValueError('Cannot set both "dev_ratio" and "folds"!')
        self.batch_size = batch_size
        self.dev_ratio = dev_ratio
        self.folds = folds
        self.stratified = stratified
        self.seed = seed
        self.n_workers = n_workers
        self.dataset = _TrainDevData(train_npz, seed, aug)

    def __call__(self):
        if self.dev_ratio:
            # train test split
            train_idx, dev_idx = self.dataset.splitter.split(
                self.dev_ratio, self.stratified)
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(train_idx),
                                      num_workers=self.n_workers)
            dev_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    sampler=_SubsetSequentialSampler(dev_idx),
                                    num_workers=self.n_workers)
            return train_loader, dev_loader
        else:
            # kfolds cv
            train_indices, dev_indices = self.dataset.splitter.kfold(
                self.folds, self.stratified)
            loaders = []
            for i in range(self.folds):
                train_loader = DataLoader(
                    self.dataset, batch_size=self.batch_size,
                    sampler=SubsetRandomSampler(train_indices[i]),
                    num_workers=self.n_workers)
                dev_loader = DataLoader(
                    self.dataset, batch_size=self.batch_size,
                    sampler=_SubsetSequentialSampler(dev_indices[i]),
                    num_workers=self.n_workers)
                loaders.append((train_loader, dev_loader))
            return loaders


class TestLoader(object):

    def __init__(self, test_npz, batch_size=1, n_workers=0, aug=False):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dataset = _TestData(test_npz, aug)

    def __call__(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          num_workers=self.n_workers)
