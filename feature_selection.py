import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import gc
from base import *
import datetime
from sklearn.model_selection import train_test_split
import sys
import warnings
import os
from base import *


def concat_file(filename, key, df, starts, stops, data):
    print(filename,key)
    path = './feature/'
    temp = pd.read_hdf(path + filename, key, start=starts, stop=stops).reset_index(drop=True)
    if data == 'train':
        temp['is_attributed'] = df['is_attributed']
        temp = down_sample(temp)
        temp.drop('is_attributed',1,inplace=True)
    for x in temp.columns:
        df[x] = temp[x]
    del temp
    gc.collect()
    return df


def get_train_data(data='train', day=8, try_test=False):
    if data == 'train':
        if day == 8:
            starts = 59709852
            stops = 122070801
        if day == 7:
            starts = 483
            stops = 59709852
        if day == 78:
            starts = 483
            stops = 122070801
        if day == 789:
            starts = 483
            stops = 184903443      

    if data == 'public':
        starts = 144708152
        stops = 148740842

    if data == 'test':
        starts = 184903890
        stops = None

    if try_test:
        stops = starts + 100000
        
    if sys.platform=='win32':
        print('begin local test')
        starts = stops = None
        
    df = pd.read_hdf('./data/data.h5', 'all', start=starts,stop=stops).reset_index(drop=True)
    df = df[df.click_id.notnull()].reset_index(drop=True)
    
    if data=='train':
        df = down_sample(df)
        
    df = concat_file('cnt.h5', '1', df, starts, stops, data)
    df = concat_file('cnt.h5', '2', df, starts, stops, data)
    df = concat_file('cnt.h5', '3', df, starts, stops, data)

    
    if data == 'private' and try_test == False:
        df = df[df['hour'].isin([13,17,18,21,22])]

    df = df.loc[:, ~df.columns.duplicated()]
    print('train rows is ',len(df))
    if data！='test':
        print(df.is_attributed.value_counts())
    return df

    
def lgb_train(train_df, val_df_v1):
    begin = datetime.datetime.now()
    params = {
        #'boosting':'dart',
        #'drop_rate':0.01,
        #'max_drop':6,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.08,
        'nthread': 32,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 15,  # 2^max_depth - 1
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 2,  # because training data is extremely unbalanced
        'num_boost_round':1000,
        'early_stopping_rounds':40,
        'verbose': -1
    }
    evals_results1 = {}
    evals_results2 = {}
    bst1 = lgb.train(
                    params,
                    train_df,
                    valid_sets=[train_df,val_df_v1],
                    valid_names=['train_df','val_df_v1'],
                    evals_result=evals_results1,
                    verbose_eval = 0
                     )
    
    
    n_estimators1 = bst1.best_iteration
    end = datetime.datetime.now()
    print((end - begin).total_seconds() // 60, ' minutes features Done')
    return bst1,evals_results1['val_df_v1']['auc'][n_estimators1 - 1],n_estimators1
    
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    begin_time = datetime.datetime.now()
    try_test = False    
    train = get_train_data(data='train',day=78)
    df_v1 = get_train_data(data='public')
    #df_v2 = get_train_data(data='private')

    test_col = \
    list(pd.read_hdf('./feature/cnt.h5', '1', stop=1).columns) + \
    list(pd.read_hdf('./feature/cnt.h5', '2', stop=1).columns) + \
    list(pd.read_hdf('./feature/cnt.h5', '3', stop=1).columns) 
    
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    
    if os.path.exists("./feature/final_feature.txt"):
        with open("./feature/final_feature.txt",'r') as f:
            train_col =[]
            train_col = f.read()
            train_col=train_col.split('\n')
    else:
        train_col =[]
        train_col.extend(['app', 'device', 'os', 'channel','hour'])
    #train_col = ['app', 'device', 'os', 'channel'] + list(set(train.columns))-list(set(test_col))
    drop_col = []
    score1 = score2 = 0
    for x in test_col:
        print('-'*10,x,'-'*10)
        train_col.append(x)

        categorical = [x for x in train_col if x in ['app', 'device', 'os', 'channel', 'hour']]
        
        train_df = lgb.Dataset(train[train_col].values.astype('float32',copy=False), label=train[target].values,feature_name=train_col, categorical_feature=categorical,free_raw_data=False)
        val_df_v1 = lgb.Dataset(df_v1[train_col].values.astype('float32',copy=False), label=df_v1[target].values,feature_name=train_col, categorical_feature=categorical,free_raw_data=False)
        
        bst1,result1,n_estimators1 = lgb_train(train_df, val_df_v1)
        
        with open("./feature/{}_feature.txt".format(str(datetime.datetime.today().day)+str(datetime.datetime.today().hour)),'a') as f:
            f.write('\n'+ str(x) + '  %.5f' % result1 + '  %.5f' %(result1 - score1))
            
        if result1 - score1 >0.00002:
            score1 = result1
        else:
            drop_col.append(x)
            train_col.remove(x)
            
        print('train feature', train_col)
        print('drop feature', drop_col)
        print(score1,result1)
        print('-'*30)
    
    test_df = get_train_data(data='test')
    id = pd.read_csv('./data/test.csv', usecols=['click_id'])
    sub = pd.DataFrame()
    sub['click_id'] = id['click_id'].astype(np.uint32)
    del id;gc.collect()
    sub['is_attributed'] = bst1.predict(test_df[train_col], num_iteration=n_estimators1)
    print("writing...")
    sub.to_csv('./sub/{}.csv.gz'.format(str(datetime.datetime.today().day)+str(datetime.datetime.today().hour)), index=False, compression='gzip')
    os.system('kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f ./sub/{}.csv.gz -m "Message"'.format(str(datetime.datetime.today().day)+str(datetime.datetime.today().hour)))
    with open(r"./feature/final_feature.txt",'w') as f:
        f.write('\n'.join(train_col))

    print('Done')
