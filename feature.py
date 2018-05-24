import gc
import os
import sys
import dask.dataframe as dd
import time, datetime
import os
from base import *
import numpy as np  # linear algebra
import pandas as pd

ori_col = ['ip', 'device', 'os', 'channel', 'app', 'hour', 'is_attributed', 'day', 'mit', 'sec', 'half_hour',
           'ten_mit_hour', 'in_test_hh', 'cate_v1', 'cate_v2', 'click_time','click_id']


def df_add_counts(df, cols):
    print('create count features ',cols)
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols) + '_count'] = reduce_series(counts[unqtags])
    del unq, unqtags, counts
    gc.collect()


def df_add_unique(df, cols):
    col = cols[1]
    cols = cols[0]
    print('create unique features ',cols, col)
    df["_".join(cols) + '__' + col + '_unique'] = reduce_series(df[cols + [col]].groupby(cols)[col].nunique().reindex(
        df.set_index(cols).index).values)


def df_add_ratio(df, cols1, cols2):
    print('create ratio features ',cols1,cols2)
    df["_".join(cols1)+"/".join(cols2)] = reduce_series(np.log1p(df["_".join(cols1) + '_count'])-np.log1p(df["_".join(cols2) + '_count']))


def df_add_var(df, cols, col):
    print('create var features ',cols, col)
    df["_".join(cols) + '__' + col + '_var'] = reduce_series(df[cols + [col]].groupby(cols)[col].var().reindex(
        df.set_index(cols).index).values)
        

def df_add_time(df, cols):
    print('create next_click features ',cols)
    df["_".join(cols) + '_nextClick'] = reduce_series((df.groupby(cols).click_time.shift(-1) - df.click_time).astype(np.float32).values)
    df["_".join(cols) + '_next2Click'] = reduce_series((df.groupby(cols).click_time.shift(-2) - df.click_time).astype(np.float32).values)
    df["_".join(cols) + '_preClick'] = reduce_series((df.groupby(cols).click_time.shift(+1) - df.click_time).astype(np.float32).values)
