#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import pickle


# In[31]:


test_sub=pd.read_csv('../user_data/lgb_sub_prob.csv')
with open('../user_data/nn_preds_0520.pkl','rb')  as f:
    nn_prob=pickle.load(f)


# In[32]:


test_sub['lgb_pred_0']=test_sub['lgb_pred_0']*0.3+nn_prob[:,0]*0.7
test_sub['lgb_pred_1']=test_sub['lgb_pred_1']*0.3+nn_prob[:,1]*0.7
test_sub['lgb_pred_2']=test_sub['lgb_pred_2']*0.3+nn_prob[:,2]*0.7
test_sub['label']=np.argmax(test_sub[['lgb_pred_0','lgb_pred_1','lgb_pred_2']].values,axis=1)+1


# In[33]:


print('final_sub',test_sub['label'].value_counts())


# In[34]:


test_sub[['linkid','current_slice_id','future_slice_id','label']].to_csv('../prediction_result/result.csv',index=False)


# In[ ]:


print('merge success!!!')

