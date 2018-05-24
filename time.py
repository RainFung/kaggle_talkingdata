import os
import pandas as pd
import time
import numpy as np
from base import *
import gc
from feature import *

def create_time_hdf(use_columns,features,filename,test=False):
    df = get_data(data='all', columns=use_columns)
    df = pd.concat([df[:184903890].sort_values(by=['click_time','is_attributed']),df[184903890:]])
    ori_col = list(df.columns)
    begin = datetime.datetime.now()
    for x in features:
        df_add_time(df, x)
    df = df.sort_index()
    df = df[df.click_id.notnull()][[x for x in df.columns if x not in ori_col]]
    print(df.columns)
    df.to_hdf('./feature/time.h5', filename, complevel=5)
    end = datetime.datetime.now()
    print('feature number is '+ str(len(df.columns)))
    print((end - begin).total_seconds() // 60, 'minutes {} time features Done'.format(filename))
    print('-'*100)
    
        
if __name__=='__main__':
    base_feature = ['app','ip', 'device', 'os', 'channel']
    features = []
    features.append(['cate_v1', 'app'])
    features.append(['cate_v1', 'channel'])
    features.append(['cate_v1', 'app', 'channel'])
    create_time_hdf(['app', 'ip', 'device', 'os', 'channel', 'click_id','cate_v1','click_time','is_attributed'], features, '1')
#------------------------------------------------------------------------------------------------------------------------------------------
    base_feature = ['app','ip', 'device', 'os', 'channel']
    features = []
    for i in range(0,5):
        for j in range(i+1,5):
            features.append([base_feature[i],base_feature[j]])
    create_time_hdf(['app', 'ip', 'device', 'os', 'channel', 'click_id','cate_v1','click_time','is_attributed'], features, '2')
#------------------------------------------------------------------------------------------------------------------------------------------
    base_feature = ['app','ip', 'device', 'os', 'channel']
    features = []
    for i in range(0,5):
        for j in range(i+1,5):
            for k in range(j+1,5):
                features.append([base_feature[i], base_feature[j], base_feature[k]])
    create_time_hdf(['app', 'ip', 'device', 'os', 'channel', 'click_id','cate_v1','click_time','is_attributed'], features, '3')
#------------------------------------------------------------------------------------------------------------------------------------------

