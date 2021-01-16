import pandas as pd
from tqdm import tqdm

'''
Convert the or_data to dataFrame format
'''
def get_data(path):
    columns=['link','label','current_slice_id','future_slice_id','recent_feature_1','recent_feature_2','recent_feature_3'
                  ,'recent_feature_4','recent_feature_5']
    for i in range(1,5):
        for j in range(1,6):
            columns.append('history_feature_cycle{}_gap{}'.format(i,j))
    data=pd.read_csv(path,header=None,sep=';')
    print(data.shape)
    'id'
    data['temp']=data[0].apply(lambda x: x.split())
    for i in range(4):
        data[columns[i]]=data['temp'].apply(lambda x: x[i])
    'recent features'
    data['temp']=data[1].apply(lambda x: x.split())
    for i in range(4,9):
        data[columns[i]]=data['temp'].apply(lambda x: x[i-4])
    'his features1'
    data['temp']=data[2].apply(lambda x: x.split())
    for i in range(9,9+5):
        data[columns[i]]=data['temp'].apply(lambda x: x[i-9])
    'his features2'
    data['temp']=data[3].apply(lambda x: x.split())
    for i in range(14,14+5):
        data[columns[i]]=data['temp'].apply(lambda x: x[i-14])
    'his features3'
    data['temp']=data[4].apply(lambda x: x.split())
    for i in range(19,19+5):
        data[columns[i]]=data['temp'].apply(lambda x: x[i-19])
    'his features4'
    data['temp']=data[5].apply(lambda x: x.split())
    for i in range(24,24+5):
        data[columns[i]]=data['temp'].apply(lambda x: x[i-24])
    data=data[columns]
    data_columns=data.columns
    save_columns=[]
    for col in tqdm(data_columns):
        if 'feature' in col:
            data['temp']=data[col].apply(lambda x:x.split(','))
            data[col+'_speed']=data['temp'].apply(lambda x: float(x[0].split(':')[1]))
            data[col+'_eta']=data['temp'].apply(lambda x: float(x[1]))
            data[col+'_status']=data['temp'].apply(lambda x: float(x[2]))
            data[col+'_num_car']=data['temp'].apply(lambda x: float(x[3]))
            save_columns.extend([col+'_speed',col+'_eta',col+'_status',col+'_num_car'])
    data=data[columns[:4]+save_columns]
    return data

import pandas as pd
import numpy as np
import lightgbm as lgb
import  torch
import scipy
import  sklearn
import  logging

logging.warning(u"get or data success!!!")
logging.warning(u"assert env:")
logging.warning("------------------------------")
logging.warning(u"pd.__version__:{}".format(pd.__version__))
logging.warning(u"sklearn.__version__{}".format(sklearn.__version__))
logging.warning(u'numpy version{}'.format(np.__version__))
logging.warning(u'lgb version{}'.format(lgb.__version__))
logging.warning(u'torch version{}'.format(torch.__version__))
logging.warning(u'torch.cuda.is_available{}'.format(torch.cuda.is_available()))
logging.warning(u'scipy.__version__{}'.format(scipy.__version__))
logging.warning("------------------------------")

if __name__ == '__main__':


    or_data_path='../../user_data/traffic/{}.txt'  # train data path
    save_data_path='../../user_data/train_{}.pkl'  # save train data path
    test_data_path='../../raw_data/20190801_testdata.txt'  # test data path
    test_save_data_path='../../user_data/test_new.pkl'  # save test  data path

    for i in range(1, 31):
        #for debug
        print(i)
        num = 20190700 + i
        train_df = get_data(or_data_path.format(num))
        train_df['day'] = i
        print(train_df.shape)
        train_df.to_pickle(save_data_path.format(i))

    test = get_data(test_data_path)
    test['day'] = 32
    test.to_pickle(test_save_data_path)

