from __future__ import print_function
import numpy as np
import pandas as pd
import cv2


data_dir = 'dataset/'


def process_band(band):
    # list to numpy 2d array
    img = np.array(band).reshape((75,75))
    return img


def pack_imgs(df):
    samples = df.shape[0]
    w, h = df['band_1'][0].shape
    img_buf = np.zeros((samples, w, h, 2), dtype=np.float32)
    for i in range(samples):
        img_buf[i, ..., 0] = df.iloc[i, df.columns.get_loc('band_1')]
        img_buf[i, ..., 1] = df.iloc[i, df.columns.get_loc('band_2')]
    return img_buf

print('reading train data...')
df_train = pd.read_json(data_dir + 'train.json')
df_train['band_1'] = df_train['band_1'].apply(process_band)
df_train['band_2'] = df_train['band_2'].apply(process_band)

print('reading test data...')
df_test = pd.read_json(data_dir + 'test.json')
df_test['band_1'] = df_test['band_1'].apply(process_band)
df_test['band_2'] = df_test['band_2'].apply(process_band)

print('normalizing...')
minv = min(df_train['band_1'].apply(np.min).min(),
           df_train['band_2'].apply(np.min).min())
maxv = max(df_train['band_1'].apply(np.max).max(),
           df_train['band_2'].apply(np.max).max())

df_train['band_1'] = df_train['band_1'].apply(
    lambda x: 2*(x-minv)/(maxv-minv)-1)
df_train['band_2'] = df_train['band_2'].apply(
    lambda x: 2*(x-minv)/(maxv-minv)-1)
df_test['band_1'] = df_test['band_1'].apply(
    lambda x: 2*(x-minv)/(maxv-minv)-1)
df_test['band_2'] = df_test['band_2'].apply(
    lambda x: 2*(x-minv)/(maxv-minv)-1)

file_train = data_dir + 'train.npz'
file_test = data_dir + 'test.npz'

print('saving train data...')
img = pack_imgs(df_train)
np.savez(file_train, img=img, y_train=df_train['is_iceberg'])

print('saving test data...')
img = pack_imgs(df_test)
np.savez(file_test, img=img, ID=df_test['id'].values)
