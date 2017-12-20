import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold


LOG = logging.getLogger(__name__)


class AugImgBase(object):
    RANDOM = -1

    def __init__(self, mode=RANDOM):
        self.mode = mode

    def __call__(self, img):
        raise NotImplementedError


class AugImgFlip(AugImgBase):
    VERTICAL = 1
    HORIZONTAL = 2
    BOTH = 3

    # img shape: H * W * C
    def __call__(self, img):
        mode = self.mode
        if mode == AugImgBase.RANDOM:
            mode = np.random.randint(0, 4)
        if mode & 1:
            img = np.flip(img, 0)
        if mode & 2:
            img = np.flip(img, 1)
        return img


class AugImgRot90(AugImgBase):
    ROT90 = 1
    ROT180 = 2
    ROT270 = 3

    # img shape: H * W * C
    def __call__(self, img):
        assert(img.shape[0] == img.shape[1])
        mode = self.mode
        if mode == AugImgBase.RANDOM:
            mode = np.random.randint(0, 4)
        return np.rot90(img, mode)


class ImgToTensor(object):

    def __call__(self, img):
        # H * W * C --> C * H * W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img.copy())


class StatoilTrainData(Dataset):
    """ Holds statoil train dataset """

    def __init__(self, train_npz):
        data = np.load(train_npz)
        self.X = data['img'].astype(np.float32)
        self.y = data['y_train'].astype(np.float32)[..., None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StatoilTestData(Dataset):
    """ Holds statoil test dataset """

    def __init__(self, test_npz):
        data = np.load(test_npz)
        self.X = data['img'].astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StatoilDataStride(Dataset):
    """ Interface to DataLoader. Reindex to actual datasets. """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.totensor = ImgToTensor()
        self.aug = []

    def _set_augment(self, aug):
        if isinstance(aug, list):
            self.aug = aug
        else:
            self.aug = [aug]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        X, y = self.dataset[self.indices[idx]]
        for aug in self.aug:
            X = aug(X)
        X = self.totensor(X)
        return X, y


class StatoilTrainSplitter(object):
    """ Split dataset for train and validation, or cross validataion """

    def __init__(self, dataset):
        self.dataset = dataset
        self.y = dataset.y.squeeze()
        self.X = np.zeros(len(self.y))

    def split(self, dev_ratio=0.2, stratified=True, seed=None):
        CVClass = StratifiedShuffleSplit if stratified else ShuffleSplit
        cv = CVClass(test_size=dev_ratio, random_state=seed)
        train_index, dev_index = next(cv.split(self.X, self.y))
        train_stride = StatoilDataStride(self.dataset, train_index)
        dev_stride = StatoilDataStride(self.dataset, dev_index)
        return train_stride, dev_stride
 
    def kfold(self, folds=5, stratified=True, seed=None):
        KFClass = StratifiedKFold if stratified else KFold
        kf = KFClass(n_splits=folds, shuffle=True, random_state=seed)
        for train_index, dev_index in kf.split(self.X, self.y):
            train_stride = StatoilDataStride(self.dataset, train_index)
            dev_stride = StatoilDataStride(self.dataset, dev_index)
            yield train_stride, dev_stride


class StatoilTrainLoader(object):

    def __init__(self, train_npz, batch_size=64, dev_ratio=None, folds=None,
                 stratified=True, seed=None, n_workers=0, train_aug=True):
        if dev_ratio is None and folds is None:
            raise ValueError('Either "dev_ratio" or "folds" must be set')
        elif dev_ratio and folds:
            raise ValueError('Cannot set both "dev_ratio" and "folds"')
        self.batch_size = batch_size
        self.dev_ratio = dev_ratio
        self.folds = folds
        self.stratified = stratified
        self.seed = seed
        self.n_workers = n_workers
        self.train_aug = [AugImgFlip(), AugImgRot90()] if train_aug else []

        dataset = StatoilTrainData(train_npz)
        self.splitter = StatoilTrainSplitter(dataset)

    def __call__(self):
        if self.dev_ratio:
            # train test split
            train_stride, dev_stride = self.splitter.split(self.dev_ratio,
                                                           self.stratified,
                                                           self.seed)
            train_stride._set_augment(self.train_aug)
            train_loader = DataLoader(train_stride, shuffle=True,
                                      batch_size=self.batch_size,
                                      num_workers=self.n_workers)
            dev_loader = DataLoader(dev_stride,
                                    batch_size=self.batch_size,
                                    num_workers=self.n_workers)
            yield train_loader, dev_loader
        else:
            # kfolds cv
            strides = self.splitter.kfold(self.folds, self.stratified,
                                          self.seed)
            for train_stride, dev_stride in strides:
                train_stride._set_augment(self.train_aug)
                train_loader = DataLoader(train_stride, shuffle=True,
                                          batch_size=self.batch_size,
                                          num_workers=self.n_workers)
                dev_loader = DataLoader(dev_stride,
                                        batch_size=self.batch_size,
                                        num_workers=self.n_workers)
                yield train_loader, dev_loader


class StatoilTestLoader(object):

    def __init__(self, test_npz, batch_size=64, n_workers=0, test_aug=False):
        self.batch_size = batch_size
        self.n_workers = n_workers
        dataset = StatoilTestData(test_npz)
        if test_aug:
            raise NotImplementedError
        else:
            self.stride = StatoilDataStride(dataset, np.arange(len(dataset)))

    def __call__(self):
        return StatoilDataLoader(self.stride, batch_size=self.batch_size,
                                 num_workers=self.n_workers)
