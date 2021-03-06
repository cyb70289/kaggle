import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold


LOG = logging.getLogger(__name__)

LOGLEVEL = (('debug', logging.DEBUG),
            ('info', logging.INFO),
            ('warn', logging.WARN),
            ('error', logging.ERROR))


############################################################
# image augmentation
# - imgae shape: H x W x C
############################################################

class StatoilAugRot90(object):

    def __init__(self, mode=None):
        """ mode = 1,2,3 -> rotate 90,180,270 """
        self.mode = mode

    # img shape: H * W * C
    def __call__(self, img):
        mode = self.mode
        if mode is None:
            mode = np.random.randint(0, 4)
        return np.rot90(img, mode)


class StatoilAugRotate(object):

    def __init__(self, angle=(-15, 15)):
        self.angle = angle

    def __call__(self, img):
        angle = np.random.randint(*self.angle)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img2 = cv2.warpAffine(img, M, (w, h), borderValue=-999)
        # fix border
        idx = img2 == -999
        img2[idx] = img[idx]
        return img2


class StatoilAugResize(object):

    def __init__(self, ratio=(0.9, 1)):
        self.ratio = ratio

    def __call__(self, img):
        ratio = np.random.rand()
        ratio *= (self.ratio[1] - self.ratio[0])
        ratio += self.ratio[0]
        h, w = img.shape[:2]
        h2, w2 = int(h*ratio), int(w*ratio)
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        img2 = cv2.resize(img, (w2, h2), interpolation=interp)
        # fit to original size
        hgap, wgap = (h-h2)//2, (w-w2)//2
        if hgap > 0:
            img3 = img.copy()
            img3[hgap:hgap+h2, wgap:wgap+w2] = img2
        elif hgap < 0:
            img3 = img2[-hgap:-hgap+h, -wgap:-wgap+w]
        else:
            img3 = img
        return img3


class StatoilAugShift(object):

    def __init__(self, delta=(6,6)):
        self.deltah, self.deltav = delta

    def __call__(self, img):
        augimg = np.copy(img)
        dh = np.random.randint(-self.deltah, self.deltah+1)
        dv = np.random.randint(-self.deltav, self.deltav+1)
        # shift horizontal
        if dh > 0:
            augimg[:, dh:] = img[:, :-dh]
        elif dh < 0:
            augimg[:, :dh] = img[:, -dh:]
        # shift vertical
        if dv > 0:
            augimg[dv:, :] = img[:-dv, :]
        elif dv < 0:
            augimg[:dv, :] = img[-dv:, :]
        return augimg


class StatoilAugFlip(object):
    """ only flip over horizontal axis """

    def __call__(self, img):
        if np.random.randint(0, 2) == 1:
            img = np.flip(img, 0)
        return img


############################################################
# dataset and dataloader
############################################################

class StatoilTrainData(Dataset):
    """ Holds statoil train dataset """

    def __init__(self, train_npz):
        data = np.load(train_npz)
        self.X_img = data['img']
        self.y = data['label'][..., None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_img[idx], self.y[idx]


class StatoilTestData(Dataset):
    """ Holds statoil test dataset """

    def __init__(self, test_npz):
        data = np.load(test_npz)
        self.X_img = data['img']

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        # add dummy label to be compatible with train data
        return self.X_img[idx], 0


class StatoilDataStride(Dataset):
    """ Interface to DataLoader. Reindex to actual datasets. """

    def __init__(self, dataset, indices, aug=[]):
        self.dataset = dataset
        self.indices = indices
        self._set_augment(aug)

    def _set_augment(self, aug):
        if isinstance(aug, list):
            self.aug = aug
        else:
            self.aug = [aug]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        X_img, y = self.dataset[self.indices[idx]]
        for aug in self.aug:
            X_img = aug(X_img)
        # H * W * C --> C * H * W
        X_img = X_img.transpose((2, 0, 1))
        X_img = torch.from_numpy(X_img.copy())
        return X_img, y


class StatoilTrainSplitter(object):
    """ Split dataset for train and validation, or cross validataion.
        One stride attachs to one DataLoader.
    """

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
        self.train_aug = []
        if train_aug:
            self.train_aug = [StatoilAugFlip(), StatoilAugRot90(),
                              StatoilAugShift()]

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
                dev_loader = DataLoader(dev_stride, shuffle=False,
                                        batch_size=self.batch_size,
                                        num_workers=self.n_workers)
                yield train_loader, dev_loader


class StatoilTestLoader(object):

    def __init__(self, test_npz, batch_size=64, n_workers=0, test_aug=False):
        self.batch_size = batch_size
        self.n_workers = n_workers
        dataset = StatoilTestData(test_npz)

        indices = np.arange(len(dataset))
        self.strides = []
        self.strides.append(StatoilDataStride(dataset, indices))
        if test_aug:
            self.strides.append(StatoilDataStride(dataset, indices,
                                                  StatoilAugRot90(1)))
            self.strides.append(StatoilDataStride(dataset, indices,
                                                  StatoilAugRot90(2)))
            self.strides.append(StatoilDataStride(dataset, indices,
                                                  StatoilAugRot90(3)))

    def __call__(self):
        loaders = []
        for stride in self.strides:
            loader = DataLoader(stride, batch_size=self.batch_size,
                                num_workers=self.n_workers, shuffle=False)
            loaders.append(loader)
        return loaders


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

    def update(self, loss):
        pass


class LRSchedStep(LRSchedNone):
    """ Learning rate scheduler based on predefine (loss, lr) pairs """

    def __init__(self, param_groups, lr, *steps):
        super(LRSchedStep, self).__init__(param_groups, lr)
        self.steps = steps

    def update(self, loss):
        for step_loss, step_lr in self.steps:
            # update lr if loss below threshold
            if loss < step_loss and self.lr > step_lr:
                self.set_lr(step_lr)
                LOG.info('Update learning rate to {:.5f}'.format(step_lr))


class LRSchedDecay(LRSchedNone):
    """ Learning rate decay on each epoch """

    def __init__(self, param_groups, lr, decay, lr_min=0.0):
        super(LRSchedDecay, self).__init__(param_groups, lr)
        self.decay = decay
        self.lr_min = lr_min

    def update(self, loss):
        if self.lr > self.lr_min:
            lr = max(self.lr*self.decay, self.lr_min)
            self.set_lr(lr)
            LOG.debug('Update learning rate to {:.5f}'.format(lr))
