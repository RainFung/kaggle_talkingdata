import gc
import os
import sys
import dask.dataframe as dd
import time, datetime
import os
import pandas as pd
import time, datetime
import numpy as np
import gc


def reduce_df(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    # print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)
                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint) ** 2
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # print("******************************")
            # print("Column: ", col)
            # print("dtype before: ", props[col].dtype)
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint32)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
                        # Make float datatypes
            else:
                if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:
                    props[col] = props[col].astype(np.float16)
                elif mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                    props[col] = props[col].astype(np.float32)
                elif mn > np.finfo(np.float64).min and mx < np.finfo(np.float64).max:
                    props[col] = props[col].astype(np.float64)
                    # print("dtype after: ", props[col].dtype)
                    # print("******************************")
    # Print final result
    # print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    # print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props


def reduce_series(props):
    props = pd.DataFrame(props)
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    # print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)
                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint) ** 2
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # print("******************************")
            # print("Column: ", col)
            # print("dtype before: ", props[col].dtype)
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint32)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
                        # Make float datatypes
            else:
                if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:
                    props[col] = props[col].astype(np.float16)
                elif mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                    props[col] = props[col].astype(np.float32)
                elif mn > np.finfo(np.float64).min and mx < np.finfo(np.float64).max:
                    props[col] = props[col].astype(np.float64)
                    # print("dtype after: ", props[col].dtype)
                    # print("******************************")
    # Print final result
    # print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    # print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props.values


# 储存数据集，设置小数据测试
def get_data(data='all', test=False, columns = None):
    end = None
    if test:
        end = 100000
    if data == 'all':
        df = pd.read_hdf('./data/data.h5', 'all', stop=end)
    if columns != None:
        df = df[columns]
    return df


def down_sample(df):
    neg = df[df['is_attributed'] == 0].sample(frac=0.02, random_state=2018)
    pos = df[df['is_attributed'] == 1]
    del df;gc.collect()
    df = pd.concat([neg,pos]).reset_index(drop=True)
    return df

