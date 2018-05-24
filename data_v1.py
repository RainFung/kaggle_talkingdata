from base import *
import pandas as pd
import numpy as np
import gc


def group(df, cols, drop=False):
    # Collapse all categorical features into a single feature
    # process = psutil.Process(os.getpid())
    # print( imax, amax, dmax, omax, cmax )
    i = 0
    for i in range(0, len(cols)):
        if i == 0:
            df['category'] = df[cols[i]].values.astype('int64')
            if drop:
                df.drop(cols[i], axis=1, inplace=True)
        else:
            df['category'] *= df[cols[i]].max()
            df['category'] += df[cols[i]]
            if drop:
                df.drop([cols[i]], axis=1, inplace=True)

    gc.collect()
    # Replace values for combined feature with a group ID, to make it smaller
    # print('\nGrouping by combined category...')
    df['category'] = df.groupby(['category']).ngroup().astype('uint32')
    gc.collect()
    # print('Total memory in use after categorizing train: ', process.memory_info().rss/(2**30), ' GB\n')
    #######  SORT BY CATEGORY AND INDEX  #######
    # Collapse category and index into a single column
    df['category'] = df.category.astype('int64').multiply(2 ** 32).add(df.index.values.astype('int32'))
    gc.collect()

    # Sort by category+index (leaving each category separate, sorted by index)
    print('\nSorting...')
    df = df.sort_values(['category'])
    gc.collect()

    # Retrieve category from combined column
    df['category'] = df.category.floordiv(2 ** 32).astype('int32')
    gc.collect()
    # print('Total memory in use after sorting: ', process.memory_info().rss/(2**30), ' GB\n')
    return df

test_sup = pd.read_csv('./data/test_supplement.csv')
print(len(test_sup))
mapping = pd.read_csv('./data/mapping.csv', dtype={'click_id': 'int32', 'old_click_id': 'int32'})
print(len(mapping))
test_sup.rename(columns={'click_id': 'old_click_id'}, inplace=True)
test_sup = pd.merge(test_sup, mapping, on=['old_click_id'], how='outer')
del mapping;gc.collect()
test_sup.drop(['old_click_id'], axis=1, inplace=True)
#test_sup['click_id'] = test_sup['click_id'].astype(np.int32)
test_sup.rename(columns={'click_id': 'sup_click_id'}, inplace=True)
print(len(test_sup))

test = pd.read_csv('./data/test.csv', usecols=['click_id'])
test = pd.merge(test, test_sup, how='outer', left_on='click_id',right_on='sup_click_id').drop('sup_click_id',1)
test = reduce_df(test)
del test_sup;gc.collect()
print(test.head(10))
print(test.tail(10))
#-----------------------------------------------------------------------------------------------------------------------------------------
train = pd.read_csv('./data/train.csv')
train = reduce_df(train)
train['click_id'] = -1
data = pd.concat([train,test]).reset_index(drop=True)
del train,test;gc.collect()
data['click_time'] = pd.to_datetime(data['click_time'])+pd.to_timedelta(8,unit='h')
data['day'] = reduce_series(data['click_time'].dt.day)
data['hour'] = reduce_series(data['click_time'].dt.hour)
data['mit'] = reduce_series(data['click_time'].dt.minute)



data = group(data,['ip','device','os'])
data.rename(columns={'category':'user'},inplace=True)
data.sort_index(inplace=True)


data['in_test_hh'] = (3 - 2 * (data['hour'].isin([12,13,17,18,21,22])) - 1 * data['hour'].isin([14,19,23]))
data['hour_mit'] = data['hour']*60 + data['mit']
data['click_time'] = (data['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)

del data['attributed_time']
data = reduce_df(data)
print(data.columns)
data.to_hdf('./data/data.h5','all',complevel=5)
print('Done' +'!'*30)
