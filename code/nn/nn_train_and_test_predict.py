#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
tqdm.pandas()

import sys
import gc
import collections
import datetime

###print
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all" 

import warnings
warnings.filterwarnings("ignore")

###torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from  torch.optim import *


# In[2]:


#### pandas_reduce_mem_usage
def pandas_reduce_mem_usage(df,igore_columns=[]):
    start_mem=df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    starttime = datetime.datetime.now()
    for col in df.columns:
        if col in igore_columns:
            continue
        col_type=df[col].dtype   #每一列的类型
        if col_type !=object:    #不是object类型
            c_min=df[col].min()
            c_max=df[col].max()
            # print('{} column dtype is {} and begin convert to others'.format(col,col_type))
            if str(col_type)[:3]=='int':
                #是有符号整数
                if c_min<0:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min >= np.iinfo(np.uint8).min and c_max<=np.iinfo(np.uint8).max:
                        df[col]=df[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
            #浮点数
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # print('\t\tcolumn dtype is {}'.format(df[col].dtype))

        #是object类型，比如str
        else:
            # print('\t\tcolumns dtype is object and will convert to category')
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    endtime = datetime.datetime.now()
    print('consume times: {:.4f}'.format((endtime - starttime).seconds))
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED=2020
seed_everything(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE is:',DEVICE)


# # some path

# In[4]:


train_data_path='../../user_data/train_{}.pkl'
test_data_path='../../user_data/test_new.pkl'
attr_path='../../raw_data/attr.txt'
topo_path='../../raw_data/topo.txt'
model_save_path='../../user_data/11_26_nn_v3_weight_loss_offline.pkl'
nn_result_save_path='../../user_data/nn_sub.csv'
prob_save_path='../../user_data/nn_preds_0520.pkl'


# # load data

# In[5]:


##### load train 
train_list=[]
for i in range(1,31):
    print(i)
    t=pd.read_pickle(train_data_path.format(i))
    train_list.append(t)
    print(train_list[-1].shape)
train=pd.concat(train_list,axis=0,sort=True)
del train_list,t
gc.collect()
##### load test
test=pd.read_pickle(test_data_path)
##### change some type
cols=['link','label','current_slice_id','future_slice_id']
for col in cols:
    train[col]=train[col].astype(int)
    test[col]=test[col].astype(int)


# In[6]:


##### load attr
attr_df=pd.read_csv(attr_path,sep='\t',header=None)
attr_df.columns=['link','length','direction','path_class','speed_class','LaneNum','speed_limit','level','width']
#####scale some feat
attr_df['width']=(attr_df['width'].values-np.mean(attr_df['width'].values))/np.std(attr_df['width'].values)
attr_df['length']=(attr_df['length'].values-np.mean(attr_df['length'].values))/np.std(attr_df['length'].values)
attr_df['speed_limit']=(attr_df['speed_limit'].values-np.mean(attr_df['speed_limit'].values))/np.std(attr_df['speed_limit'].values)
train=train.merge(attr_df,on='link',how='left')
test=test.merge(attr_df,on='link',how='left')


# In[ ]:





# # get features

# ## 一些类别特征

# In[7]:


##### cate: day week_day hour time_gap
test['day']=32
train['week_day']=train['day'].apply(lambda x:x%7)
test['week_day']=4

train['hour']=train['future_slice_id'].apply(lambda x:x//30 if x>=0 else (720+x)//30)
test['hour']=test['future_slice_id'].apply(lambda x:x//30 if x>=0 else (720+x)//30)

train['time_gap']=list(map(lambda x,y: x-y if y>=0 else x-y,train['future_slice_id'], train['current_slice_id']))
test['time_gap']=list(map(lambda x,y: x-y if y>=0 else x-y,test['future_slice_id'], test['current_slice_id']))


# ## 获取图拓扑

# In[ ]:


##获取下一个链接
topo_df=pd.read_csv(topo_path,sep='\t',header=None)

topo_df.columns=['link','next_link']
topo_df['next_link']=topo_df['next_link'].apply(lambda x: [ int(i) for i in x.split(',')])
all_topo_link=set(list(topo_df['link'].values))
len(all_topo_link)
for _,row in tqdm(topo_df.iterrows()):
    next_link=row['next_link']
    for link in next_link:
        if link not in all_topo_link:all_topo_link.add(link)
len(all_topo_link)


# In[ ]:


train=train.merge(topo_df,on='link',how='left')


# In[ ]:


test=test.merge(topo_df,on='link',how='left')


# In[ ]:





# ## 特征-->id

# In[ ]:


col_thre_dict={'link':0.5, 'direction':0.5, 'path_class':0.5, 'speed_class':0.5, 'LaneNum':0.5, 'level':0.5, 'continuous_width':0.5,
               'continuous_length':0.5,'continuous_speed_limit':0.5,'time_gap':0,
              'current_slice_id':0.5,'future_slice_id':0.5,'week_day':0,'hour':0}
len(col_thre_dict)

ids_indexs={}
mp_col_ids_indexs={}
ids_indexs['padding']=0
for col,thre in tqdm(col_thre_dict.items()):
    mp={}
    unknow=None
    #连续特征一个field
    if 'continuous_' in col:
        mp[col]=len(ids_indexs)
        ids_indexs[col]=len(ids_indexs)
        unknow=len(ids_indexs)
        ids_indexs[col+'_unknow']=len(ids_indexs)
        mp_col_ids_indexs[col]=[mp,unknow]
        continue
    if col=='link':
        curr_len=len(ids_indexs)
        ###use attr
        for i,ids in enumerate(attr_df['link'].values):
            ids_indexs[col+'_'+str(ids)]=i+curr_len
            mp[ids]=i+curr_len
        if thre!=0:
            unknow=len(ids_indexs)
            ids_indexs[col+'_unknow']=len(ids_indexs)
        mp_col_ids_indexs[col]=[mp,unknow]
        continue
    t=train[col].value_counts().reset_index()
    print(col+' or:',len(t))
    all_ids=t[t[col]>thre]['index'].values
    print(col+' new:',len(all_ids))
    print(col+'test or:',test[col].nunique())
    print(col+'test new:',test[test[col].isin(all_ids)][col].nunique())
    print('*'*50)
    curr_len=len(ids_indexs)
    for i,ids in enumerate(all_ids):
        ids_indexs[col+'_'+str(ids)]=i+curr_len
        mp[ids]=i+curr_len
    if thre!=0:
        unknow=len(ids_indexs)
        ids_indexs[col+'_unknow']=len(ids_indexs)
    mp_col_ids_indexs[col]=[mp,unknow]


# In[ ]:


feat_columns=[] 
feat_value_columns=[] 
for col,thre in tqdm(col_thre_dict.items()):
    col_index_name='{}_index'.format(col)
    col_value_name='{}_value'.format(col)
    feat_columns.append(col_index_name)
    feat_value_columns.append(col_value_name)
    real_col=col
    #continuous feat
    if 'continuous_' in col:
        real_col=col.replace('continuous_','')
        train[col_index_name]=mp_col_ids_indexs[col][0][col]
        train[col_value_name]=train[real_col].values
        
        test[col_index_name]=mp_col_ids_indexs[col][0][col]
        test[col_value_name]=test[real_col].values
    # cate feat
    else:
        mp=mp_col_ids_indexs[col][0]
        unknow=mp_col_ids_indexs[col][1]
        if unknow!=None:
            train[col_index_name]=train[real_col].map(mp).fillna(unknow)
            test[col_index_name]=test[real_col].map(mp).fillna(unknow)
        else:
            train[col_index_name]=train[real_col].map(mp)
            test[col_index_name]=test[real_col].map(mp)
        train[col_value_name]=1.
        test[col_value_name]=1.


# ## 获取子图拓扑

# In[ ]:





# In[ ]:


##获取上一个链接
mp=mp_col_ids_indexs['link'][0]
unknow=mp_col_ids_indexs['link'][1]
link_before_dict={}
for _,row in tqdm(topo_df.iterrows()):
    link=row['link']
    link=mp[link]
    next_link=row['next_link']
    for next_l in next_link:
        next_l=mp[next_l]
        if next_l not in link_before_dict:link_before_dict[next_l]=[]
        if link not in link_before_dict[next_l]:
            link_before_dict[next_l]=link_before_dict[next_l]+[link]


# In[ ]:


##获取下一个链接
mp=mp_col_ids_indexs['link'][0]
unknow=mp_col_ids_indexs['link'][1]
link_next_dict={}
for _,row in tqdm(topo_df.iterrows()):
    link=row['link']
    link=mp[link]
    next_link=row['next_link']
    link_next_dict[link]=[]
    for next_l in next_link:
        next_l=mp[next_l]
        link_next_dict[link]=link_next_dict[link]+[next_l]


# In[ ]:


'可以做拓扑关系的link(已经转换成id)'
train_test_link=[]
for link in train['link'].unique():
    if link in all_topo_link:
        train_test_link:train_test_link.append(mp[link])
for link in test['link'].unique():
    if link in all_topo_link:
        train_test_link.append(mp[link])
train_test_link=list(set(train_test_link))
len(train_test_link)


# In[ ]:


def get_next_link(link_id):
    '获取下游'
    next_link=link_next_dict.get(link_id,[]) #下游
    return [[link_id,next_link_id] for next_link_id in next_link],len(next_link),next_link
def get_before_link(link_id):
    '获取上游'
    before_link=link_before_dict.get(link_id,[]) #下游
    return [[link_id,before_link_id]for before_link_id in before_link],len(before_link),before_link

def add_link_set(links,link_set,add=True):
    n=0
    for link in links:
        if link not in link_set:
            if add:link_set.add(link)
            n+=1
    return link_set,n

def add_link_info(link_add_info,sub_info,num,max_number,link_set):
    for info in sub_info:
        #都包含节点
        if info[1] in link_set and info[0] in link_set:
            link_add_info.append(info)
        #都不在
        elif info[1] not in link_set and info[0]  not in link_set:
            if num>max_number-2:continue
            link_set.add(info[1])
            link_set.add(info[0])
            link_add_info.append(info)
            num+=2
        #0在
        elif info[1] not in link_set:
            if num>max_number-1:continue
            link_set.add(info[1])
            link_add_info.append(info)
            num+=1
        #1在
        elif info[0] not in link_set:
            if num>max_number-1:continue
            link_set.add(info[0])
            link_add_info.append(info)
            num+=1
    return link_add_info,num,link_set


def convert_symmetric(X):
    '转换成对称矩阵'
    X += X.T +np.eye(X.shape[0])
    return X

def normalize_adj(adj):
    D=np.diag(1/np.sqrt(np.sum(adj,axis=1)))
    adj=np.dot(D,adj)
    return adj.dot(D)
print('link_number:',len(train_test_link))
import scipy.sparse as sp
link_toop_sub_graph={} #link_id:matrix
NUM_ADD_GRAPPH=4 #添加图的范围
MAX_NUMBER=200 #最大的子图数量
nums=[]
for link_id in tqdm(train_test_link):
    link_info=[] #记录边
    link_set=set([link_id])
    num=1
    next_link=link_next_dict.get(link_id,[]) #下游
    before_link=link_before_dict.get(link_id,[]) #上游
    
    ###当前上下游link
    link_set,n=add_link_set(next_link,link_set)
    num+=n
    link_set,n=add_link_set(before_link,link_set)
    num+=n
    link_info.extend([[link_id,next_link_id] for next_link_id in next_link])
    link_info.extend([[link_id,before_link_id]for before_link_id in before_link])
    
    link_add_info=[]#其他边
    #扩展NUM_ADD_GRAPPH阶
    for graph_add in range(NUM_ADD_GRAPPH):
        if num==MAX_NUMBER:break
        next_link_next=[]
        before_link_before=[]
        #next link
        for sub_link_id in next_link:
            sub_info,sub_num,sub_next_link= get_next_link(sub_link_id)
            link_add_info,num,link_set=add_link_info(link_add_info,sub_info,num,MAX_NUMBER,link_set)
            next_link_next.extend(sub_next_link)
            
            sub_info,sub_num,sub_next_link= get_before_link(sub_link_id)
            link_add_info,num,link_set=add_link_info(link_add_info,sub_info,num,MAX_NUMBER,link_set)
            next_link_next.extend(sub_next_link)
        #before link（bug：next_link_next：应该是before_link_before 但基本无影响）
        for sub_link_id in before_link:
            sub_info,sub_num,sub_before_link=get_before_link(sub_link_id)
            link_add_info,num,link_set=add_link_info(link_add_info,sub_info,num,MAX_NUMBER,link_set)
            next_link_next.extend(sub_before_link)
            
            sub_info,sub_num,sub_before_link= get_next_link(sub_link_id)
            link_add_info,num,link_set=add_link_info(link_add_info,sub_info,num,MAX_NUMBER,link_set)
            next_link_next.extend(sub_before_link)
        
        next_link=next_link_next
        before_link=before_link_before
    
    nums.append(num)
    link_info.extend(link_add_info)#所有边
    
    
    #转换成id
    edges=np.array(link_info)
    link_mp={link_id:0}
    k=0
    for sub_link_id in set(edges.flatten()):
        if sub_link_id!=link_id:
            link_mp[sub_link_id]=k+1
            k+=1
    number=len(link_mp) #当前图的个数
    edges = np.array(list(map(link_mp.get, edges.flatten())),
                     dtype=np.int32).reshape(edges.shape)
    
    #转换成稀疏邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(number, number), dtype=np.float32).toarray()
    #获得现在的稀疏矩阵id
    link_ids=list(link_mp.keys())
    
    #paddding后转换成对称矩阵
    adj=np.pad(adj, ((0, MAX_NUMBER-number), (0, MAX_NUMBER-number)), mode='constant',constant_values=(0))
    adj=convert_symmetric(adj) #对称矩阵
    adj=(adj>0).astype(np.int)
    
    #D-1/2*A*D-1/2
    adj=normalize_adj(adj)
    #pad link_id
    link_ids=link_ids+[0]*(MAX_NUMBER-number)
    #save matrix
    link_toop_sub_graph[link_id]=(adj,link_ids)


# In[ ]:


adj


# In[ ]:


np.max(nums),np.min(nums),np.mean(nums)


# In[ ]:


pd.DataFrame({'num':nums}).describe([0.9,0.95,0.98,0.99])


# # 每个link 的静态属性特征
# 转换成矩阵后作为nn中的embedding 矩阵

# In[ ]:


attr_feat_cols=['direction', 'path_class', 'speed_class', 'LaneNum', 'level', 'width','length','speed_limit'] #后面三个是连续特征

'特征转换成id表示'
for col in attr_feat_cols[:5]:
    attr_df[col]=attr_df[col].map(mp_col_ids_indexs[col][0])
'获得embedding 矩阵'
link_embeddding_matrix_cate=attr_df[attr_feat_cols].values
link_embeddding_matrix_cate=np.concatenate([np.zeros(len(attr_feat_cols)).reshape(1,-1),link_embeddding_matrix_cate],axis=0)


# In[ ]:


del attr_df,mp_col_ids_indexs,link_next_dict,link_before_dict,topo_df
gc.collect()


# # scale (序列特征)

# In[ ]:


'scale feat'
scaled_features=[]
for col in tqdm(train.columns):
    if 'feature' in col:
        scaled_features.append(col)
len(scaled_features)
means=np.mean(train[scaled_features].values,axis=0)
stds=np.std(train[scaled_features].values,axis=0)

for i,col in tqdm(enumerate(scaled_features)):
    train.loc[:,col]=(train.loc[:,col]-means[i])/stds[i]
    test.loc[:,col]=(test.loc[:,col]-means[i])/stds[i]


# In[ ]:


'recent_feature 序列'
recent_cols=[]
for i in range(1,6):
    recent_col=[ col for col in train.columns if 'recent_feature_{}'.format(i) in col]
    recent_cols.extend(recent_col)
train['recent_split_info']=[ i.reshape(4,5).reshape(5,4) for i in train[recent_cols].values]
test['recent_split_info']=[ i.reshape(4,5).reshape(5,4) for i in test[recent_cols].values]


# In[ ]:


'history_feature 序列'
his_cols=[]
for i in range(1,6):
    his_col=[ col for col in train.columns if 'history_feature_cycle{}'.format(i) in col]
    his_cols.extend(his_col)
len(his_cols)
train['his_split_info']=[ i.reshape(4,20).reshape(20,4) for i in train[his_cols].values]
test['his_split_info']=[ i.reshape(4,20).reshape(20,4) for i in test[his_cols].values]


# In[ ]:


train['label']=train['label'].apply(lambda x: x-1 if x!=4 else 2)
test['label']=0


# In[ ]:


def get_model_data(df):
    df['category_features']=[ i for i in df[feat_columns].values]
    df['category_features_values']=[ i for i in df[feat_value_columns].values]
    return df[['category_features','category_features_values','recent_split_info','his_split_info','link_index','label']].values


# In[ ]:


train_data=get_model_data(train.iloc[:20000,:])
test_data=get_model_data(test)


# # data loader

# In[ ]:


class zy_DataSet(torch.utils.data.Dataset):
    def __init__(self,data,graph_dict=link_toop_sub_graph):
        self.data=data
        self.graph_dict=graph_dict
    def __len__(self):
        return len(self.data)
    
    def get_graph_feat(self,link_id):
        if link_id not in self.graph_dict:
            return [link_id]+[0]*(MAX_NUMBER-1),np.eye(MAX_NUMBER)
        link_graph,link_seq=self.graph_dict[link_id]
        return link_seq,link_graph
    
    def __getitem__(self,index):
        feature=self.data[index,:]
        category_index=torch.tensor(feature[0],dtype=torch.long)  #label
        category_value=torch.tensor(feature[1],dtype=torch.float32)    
        recent_split_info=torch.tensor(feature[2],dtype=torch.float32)   
        his_split_info=torch.tensor(feature[3],dtype=torch.float32)
        
        link_index=feature[4]
        link_seq,link_graph=self.get_graph_feat(link_index)
        link_seq=torch.tensor(link_seq,dtype=torch.long)
        link_graph=torch.tensor(link_graph,dtype=torch.float32)
        
        label=torch.tensor(feature[5],dtype=torch.long)
        return category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label

    def collate_fn(self,batch):
        category_index = torch.stack([x[0] for x in batch])
        category_value = torch.stack([x[1] for x in batch])
        recent_split_info = torch.stack([x[2] for x in batch])
        his_split_info = torch.stack([x[3] for x in batch])
        link_seq = torch.stack([x[4] for x in batch])
        link_graph = torch.stack([x[5] for x in batch])
        label = torch.stack([x[6] for x in batch])
        return category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

def get_loader(df,batch_size=16,train_mode=False):
    ds_df = zy_DataSet(df)
    loader = DataLoaderX(ds_df, batch_size=batch_size, shuffle=train_mode, num_workers=2, collate_fn=ds_df.collate_fn, drop_last=train_mode)
    loader.num = len(ds_df)
    return loader
    
def debug_loader(d):
    loader=get_loader(d,train_mode=True)
    for category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label in loader:
        print(category_index)
        print(category_value)
        print(recent_split_info)
        print(his_split_info)
        print(label)
        print(link_seq.size())
        print(link_graph.size())
        break


# In[ ]:


debug_loader(train_data)


# # build model

# In[ ]:


######################Bi
class Bi_interaction(torch.nn.Module):
    def __init__(self):
        super(Bi_interaction, self).__init__()

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        index = square_of_sum - sum_of_square
        return 0.5 * index
    
##################GCN
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
            super(GraphConvolution, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
    
            self.reset_parameters()
    
    def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
            support = torch.matmul(input, self.weight)
            output = torch.matmul(adj, support)
            if self.bias is not None:
                return F.relu(output + self.bias)
            else:
                return F.relu(output)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )
class GCN(torch.nn.Module):
    def __init__(self,feat_dim,K=2):
        self.K=K
        super(GCN, self).__init__()
        self.gcs=nn.ModuleList([GraphConvolution(feat_dim,feat_dim) for _ in range(K)])
    
    def forward(self,A,H):
        for k in range(self.K):
            H=self.gcs[k](H,A)
        return H


# In[ ]:


class DiDi_Model(nn.Module):
    def __init__(self,embedding_num,embedding_dim,field_dims=None):
        '''
        field_dims：the number of fileds
        embedding_num : sum of the index of all fields
        embedding_dim : the dim of embedding
        '''
        super(DiDi_Model, self).__init__()
        self.model_name = 'zy_Model'
        self.field_dims=field_dims
        self.embedding_num=embedding_num
        self.embedding_dim=embedding_dim
        
        
        #link原始特征embedding
        self.link_or_em=nn.Embedding(link_embeddding_matrix_cate.shape[0],link_embeddding_matrix_cate.shape[1])
        self.link_or_em.weight.data.copy_(torch.from_numpy(link_embeddding_matrix_cate))
        self.link_or_em.requires_grad = False
        
        self.width_em=nn.Sequential(nn.Linear(1,self.embedding_dim))
        self.length_em=nn.Sequential(nn.Linear(1,self.embedding_dim))
        self.speed_em=nn.Sequential(nn.Linear(1,self.embedding_dim))
        
        #FM的一阶
        self.first_em=nn.Embedding(self.embedding_num,1)
        #FM的二阶
        self.embdedding_seq=nn.Embedding(self.embedding_num,embedding_dim)
        self.bi = Bi_interaction()
        
        #lstm
        input_dim=4
        output_put_dim=8
        self.lstm_seq=nn.LSTM(input_dim,output_put_dim,2,batch_first=True,bidirectional=False)
        self.lstm_seq_1=nn.LSTM(input_dim,output_put_dim,2,batch_first=True,bidirectional=False)
        
        #GAE
        self.gcn=GCN(K=2,feat_dim=9*32)
        

        self.embed_output_dim = embedding_dim+output_put_dim*2+output_put_dim*2*4+9*32

        self.mlp=nn.Sequential(
                               nn.Dropout(0.5),
                               nn.Linear(self.embed_output_dim,self.embed_output_dim//2),
                               nn.BatchNorm1d(self.embed_output_dim//2),
                               nn.ReLU(True),
                               nn.Dropout(0.3),
                               nn.Linear(self.embed_output_dim//2,3))
        
    def mask_mean(self,x,mask=None):
        if mask!=None:
            mask_x=x*(mask.unsqueeze(-1))
            x_sum=torch.sum(mask_x,dim=1)
            re_x=torch.div(x_sum,torch.sum(mask,dim=1).unsqueeze(-1))
        else:
            x_sum=torch.sum(x,dim=1)
            re_x=torch.div(x_sum,x.size()[1])
        return re_x
    
    
    def mask_max(self,x,mask=None):
        if mask!=None:
            mask=mask.unsqueeze(-1)
            mask_x=x-(1-mask)*1e10
            x_max=torch.max(mask_x,dim=1)
        else:
            x_max=torch.max(x,dim=1)
        return x_max[0]

    def forward(self,category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label,is_test=False):
        
        batch_size=category_index.size()[0]
        
        ##FM二阶
        seq_em=self.embdedding_seq(category_index)*category_value.unsqueeze(2)
        x2=self.bi(seq_em)
        
        #lstm
        f,_=self.lstm_seq_1(recent_split_info)
        fmax=self.mask_max(f)
        fmean=self.mask_mean(f)
        recent_features=torch.cat([fmean,fmax],dim=1)
        
        
        his_features=[]
        for i in range(4):
            his_split_info_sample=his_split_info[:,i*5:(i+1)*5,:]
            f,_=self.lstm_seq(his_split_info_sample)
            fmax=self.mask_max(f)
            fmean=self.mask_mean(f)
            his_features.append(fmax)
            his_features.append(fmean)
        his_features=torch.cat(his_features,dim=-1)
        
        
        ###GCN
        #######node feat
        number_graph_node=link_seq.size(1)
        link_feat_link_id=self.embdedding_seq(link_seq)
        
        link_feat=self.link_or_em(link_seq)# B*number_graph_node*8
        link_feat_cate=link_feat[:,:,:5].long() #类别特征  B*number_graph_node*5
        link_feat_cate=self.embdedding_seq(link_feat_cate).view(batch_size,number_graph_node,-1)
        link_feat_width=self.width_em(link_feat[:,:,5].float().unsqueeze(2)) #类别特征
        link_feat_length=self.length_em(link_feat[:,:,6].float().unsqueeze(2)) #类别特征
        link_feat_speed=self.speed_em(link_feat[:,:,7].float().unsqueeze(2)) #类别特征
        link_feat=torch.cat([link_feat_cate,link_feat_width,link_feat_length,link_feat_speed,link_feat_link_id],dim=-1) # B*node*(dim*8)
        
        
        
        gcn_out=self.gcn(link_graph,link_feat)
        gcn_out=gcn_out[:,0,:].squeeze()
        

        
        #DNN全连接
        x2=torch.cat([x2,recent_features,his_features,gcn_out],dim=1)
        x3=self.mlp(x2)
        
        out=x3
        if not is_test:
            loss_fun=nn.CrossEntropyLoss(torch.tensor([0.1,0.3,0.6]).to(DEVICE))
            loss=loss_fun(out,label)
            return loss,F.softmax(out,dim=1)
        else:
            loss_fun=nn.CrossEntropyLoss()
            loss=loss_fun(out,label)
            return loss,F.softmax(out,dim=1)
net = DiDi_Model(field_dims=len(col_thre_dict),embedding_num=len(ids_indexs),embedding_dim=32)
print('# Model parameters:', sum(param.numel() for param in net.parameters()))


# # debug model

# In[ ]:


def debug_label(d):
    loader=get_loader(d[:10000,:],train_mode=True,batch_size=2)

    model=DiDi_Model(embedding_num=len(ids_indexs),embedding_dim=32)

    for category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label  in loader:
        print(category_index.size())
        print(category_value.size())
        print(recent_split_info.size())
        print(his_split_info.size())
        y = model(category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label,is_test=False)
        print(y)
        y = model(category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label,is_test=True)
        print(y)
        break


# In[ ]:


# debug_label(train_data)


# # train fun

# In[ ]:


from sklearn.metrics import f1_score
def metric_fn(preds,real_labels):
    ##for log
    preds=np.argmax(preds,axis=1)
    f1=f1_score(real_labels, preds,average=None)
    print(f1)
    return 0.2*f1[0]+0.2*f1[1]+0.6*f1[2]

def validation_fn(model,val_loader,is_test=False):
    model.eval()
    bar = tqdm(val_loader)
    preds=[]
    labels=[]
    weights=[]
    loss_all=[]
    for i,feat in enumerate(bar):
        category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label=(_.to(DEVICE) for _ in feat)
        loss,p= model(category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label,is_test=True)
        preds.append(p.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        loss_all.append(loss.item())
    preds=np.concatenate(preds)
    labels=np.concatenate(labels)
    if not is_test:
        score=metric_fn(preds.squeeze(),labels)
        return np.mean(loss_all),score
    else:
        return preds.squeeze()
    

def train_model(model,train_loader,val_loader,accumulation_steps=2
                ,early_stop_epochs=2,epochs=4,model_save_path='pytorch_zy_model_true.pkl'):  
    
    losses=[]
    
    ########早停
    no_improve_epochs=0
    
    ########优化器 学习率
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99),weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98) #学习率衰减
    
    train_len=len(train_loader)
    
    best_vmetric=-np.inf
    loss_ages = []
    loss_genders=[]
    for epoch in range(1,epochs+1):
        model.train()
        print(scheduler.get_lr()[0])
        bar = tqdm(train_loader)
        for i,feat in enumerate(bar):
            category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label=(_.to(DEVICE) for _ in feat)
            loss,_= model(category_index,category_value,recent_split_info,his_split_info,link_seq,link_graph,label,
                          is_test=False)
            sloss=loss
            sloss.backward()
            loss_ages.append(loss.item())
            loss_genders.append(loss.item())
            if (i+1) % accumulation_steps == 0 or (i+1)==train_len:
                optimizer.step()
                optimizer.zero_grad()
            bar.set_postfix(loss_ages=np.array(loss_ages).mean(),loss_genders=np.array(loss_genders).mean(),epoch=epoch)
#         if scheduler.get_lr()[0]>0.005:
        scheduler.step()
        #val
        val_loss,mse=validation_fn(model,val_loader)
        losses.append( 'train_loss:%.5f, score: %.5f, best score: %.5f\n val loss: %.5f\n' %
            (np.array(loss_ages).mean(),mse,best_vmetric,val_loss))
        print(losses[-1])
        if mse>=best_vmetric:
            torch.save(model.state_dict(),model_save_path)
            best_vmetric=mse
            no_improve_epochs=0
            print('improve save model!!!')
        else:
            no_improve_epochs+=1
        if no_improve_epochs==early_stop_epochs:
            print('no improve score !!! stop train !!!')
            break
    return losses


# #  train loop single

# In[ ]:


#train_data=get_model_data(train[train['day']!=30])
valid_data=get_model_data(train[train['day']==30])
test_data=get_model_data(test)


# In[ ]:


valid_y=train[train['day']==30]['label'].values
del train
gc.collect()


# In[ ]:


#train_data.shape,valid_data.shape,test_data.shape


# In[ ]:


test_loader=get_loader(test_data,batch_size=1024,train_mode=False)
#tra_loader=get_loader(train_data,batch_size=1024,train_mode=True)
valid_loader=get_loader(valid_data,batch_size=1024,train_mode=False)


# In[ ]:


model=DiDi_Model(embedding_num=len(ids_indexs),embedding_dim=32).to(DEVICE) #NFM


# In[ ]:


#losses=train_model(model,tra_loader,valid_loader,
#                         accumulation_steps=1,early_stop_epochs=3,epochs=2,
#                            model_save_path=model_save_path)
#print( losses)


# In[ ]:

print('bedgin predict:')
model.load_state_dict(torch.load(model_save_path,map_location=DEVICE))
test_loader=get_loader(test_data,batch_size=1024,train_mode=False)
preds=validation_fn(model,test_loader,is_test=True)


# In[ ]:


test['label']=np.argmax(preds,axis=1)+1
test['label'].value_counts()


# In[ ]:


preds


# In[ ]:


import scipy as sp
from functools import partial

def f1_loss(weight, y_hat, y):
    y_hat = weight*y_hat
    scores = f1_score(y, np.argmax(y_hat, axis=1), average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    return -scores
def get_weights(y_hat, y):
    size = np.unique(y).size
    loss_partial = partial(f1_loss, y_hat=y_hat, y=y)
    initial_weights = [1. for _ in range(size)]
    weights_ = sp.optimize.minimize(loss_partial, initial_weights, method='Powell')
    return weights_['x']


# In[ ]:


valid_preds=validation_fn(model,valid_loader,is_test=True)


# In[ ]:


weights=get_weights(valid_preds,valid_y)


# In[ ]:


weights


# In[ ]:


preds=weights*preds
preds


# In[ ]:


test['label']=np.argmax(preds,axis=1)+1
test[['link','current_slice_id','future_slice_id','label']].to_csv(nn_result_save_path,index=False)


# In[ ]:


print(test['label'].value_counts())


# In[ ]:


import pickle
with open(prob_save_path,'wb') as f:
    pickle.dump(preds,f)


# In[ ]:





# In[ ]:




