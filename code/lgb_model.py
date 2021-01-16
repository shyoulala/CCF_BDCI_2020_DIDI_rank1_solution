'''
2.7.16 |Anaconda, Inc.| (default, Mar 14 2019, 15:42:17) [MSC v.1500 64 bit (AMD64)]

pandas==0.24.2
numpy==1.16.2
lightgbm==2.2.1
scikit-learn==0.20.3
scipy==1.2.1

解决方案：
1. 采样
2. 特征工程：
(1) 目标编码，历史同link/future_slice_id/link+小时/link+星期/link+小时+星期的三种道路状况比例
(2) 该link+future_slice_id近期的速度、车辆数，路况等统计
(3) 该link+future_slice_id近N个时间片内各状态距离当前时间的时间间隔
(4) 提取link拓扑信息中的上游和下游，统计其对应future_slice_id前的道路状况
'''


import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import minimize
from functools import partial
import scipy as sp
import gc
warnings.filterwarnings('ignore')

def day_origin_df(path):
    df = pd.read_csv(path, header=None, sep=';', usecols=[0, 1])
    df['linkid'] = df[0].apply(lambda x: int(x.split(' ')[0]))
    df['label'] = df[0].apply(lambda x: int(x.split(' ')[1]))
    df['label'] = df['label'].apply(lambda x: 3 if x > 3 else x)
    df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
    df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))
    df['curr_state'] = df[1].apply(lambda x: int(x.split(' ')[-1].split(':')[-1].split(',')[2]))
    del df[0], df[1]
    return df

def get_his_sample(time_start):
    tmp_df = pd.DataFrame()
    for d in tqdm(range(time_start, 20190731)):
        tmp_file = pd.read_csv('../user_data/traffic/{}.txt'.format(d), header=None, sep=';')
        tmp_file['day'] = d
        tmp_file['future_slice_id'] = tmp_file[0].apply(lambda x: int(x.split(' ')[3]))
        if d != 20190730:
            tmp_file = tmp_file[(tmp_file['future_slice_id']>10) & (tmp_file['future_slice_id']<40)]
        else:
            tmp_file_1 = tmp_file[(tmp_file['future_slice_id']>10) & (tmp_file['future_slice_id']<40)]
            tmp_file_2 = tmp_file[tmp_file['future_slice_id']>=40].sample(n=50000, random_state=1)
            tmp_file = tmp_file_1.append(tmp_file_2)
        del tmp_file['future_slice_id']
        tmp_df = tmp_df.append(tmp_file)
    tmp_df.to_csv('../user_data/sample_data.csv', index=False, header=None, sep=';')


def sample_train_data(origin_train, origin_test):
    origin_train['time_diff']=origin_train['future_slice_id'] - origin_train['current_slice_id']
    origin_test['time_diff']=origin_test['future_slice_id'] - origin_test['current_slice_id']

    curr_state_dict = (origin_test['curr_state'].value_counts(normalize=True)*origin_train.shape[0]).astype(int).to_dict()
    sample_train = pd.DataFrame()
    for t, group in origin_train.groupby('curr_state'):
        if t == 0:
            sample_tmp = group
        else:
            sample_tmp = group.sample(n=curr_state_dict[t], random_state=1)
        sample_train = sample_train.append(sample_tmp)

    time_dict = (origin_test['time_diff'].value_counts(normalize=True)*sample_train.shape[0]*0.65).astype(int).to_dict()

    sample_df = pd.DataFrame()
    for t, group in sample_train.groupby('time_diff'):
        sample_tmp = group.sample(n=time_dict[t], random_state=1)
        sample_df = sample_df.append(sample_tmp)

    for j, i in enumerate(range(11, 16)):
        sample_df.loc[sample_df['time_diff']==i, 'time_diff'] =  sample_df.loc[sample_df['time_diff']==i, 'time_diff']*(10+j)
    sample_df = sample_df.sort_values('time_diff').drop_duplicates(subset=['linkid', 'future_slice_id'], keep='last')

    for j, i in enumerate(range(11, 16)):
        sample_df.loc[sample_df['time_diff']==i*(10+j), 'time_diff'] =  sample_df.loc[sample_df['time_diff']==i*(10+j), 'time_diff']/(10+j)

    # tmp_merge = origin_train.merge(origin_test[['linkid', 'current_slice_id']].drop_duplicates(), on=['linkid', 'current_slice_id'], how='inner')
    # sample_df = tmp_merge.append(sample_df)
    # sample_df = sample_df.drop_duplicates(subset=['linkid', 'future_slice_id'], keep='first')
    del sample_df['time_diff']
    return sample_df


def split_features(features, index):
    features = features.split(' ')
    mid = [f.split(':')[-1] for f in features]
    result = [float(f.split(',')[index]) for f in mid]
    return result


def split_slice(features):
    features = features.split(' ')
    result = [int(f.split(':')[0]) for f in features]
    return result


def load_traffic_data(df):
    df.columns = ['0', 'recent_feature', 'history_feature_28', 'history_feature_21', 'history_feature_14', 'history_feature_7', 'day']
    df['linkid'] = df['0'].apply(lambda x: int(x.split(' ')[0]))
    df['current_slice_id'] = df['0'].apply(lambda x: int(x.split(' ')[2]))
    df['future_slice_id'] = df['0'].apply(lambda x: int(x.split(' ')[3]))

    df['recent_speed'] = df['recent_feature'].apply(lambda x: split_features(x, 0))
    df['recent_eta'] = df['recent_feature'].apply(lambda x: split_features(x, 1))
    df['recent_status'] = df['recent_feature'].apply(lambda x: split_features(x, 2))
    df['recent_vichles_num'] = df['recent_feature'].apply(lambda x: split_features(x, 3))
    df['recent_slices'] = df['recent_feature'].apply(lambda x: split_slice(x))

    for i in [28, 21, 14, 7]:
        df['history_speed_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 0))
        df['history_eta_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 1))
        df['history_status_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(lambda x: split_features(x, 2))
        df['history_vichles_num_{}'.format(i)] = df['history_feature_{}'.format(i)].apply(
            lambda x: split_features(x, 3))

    df['weekday'] = pd.to_datetime(df['day'].astype(str)).dt.weekday + 1
    df['hour'] = df['future_slice_id'].apply(lambda x: int(x / 30))
    del df['0']
    return df

def gen_ctr_features(d, key):
    his_ = his_df[his_df['day']<d].copy()
    his_ = his_.drop_duplicates(subset=['link_id', 'future_slice_id', 'day', 'label'], keep='last')
    dummy = pd.get_dummies(his_['label'], prefix='label')
    his_ = pd.concat([his_, dummy], axis=1)
    ctr = his_.groupby(key)[['label_0', 'label_1', 'label_2']].mean().reset_index()
    ctr.columns = key+['_'.join(key)+'_label_0_ctr', '_'.join(key)+'_label_1_ctr', '_'.join(key)+'_label_2_rt']
    ctr['day'] = d
    del his_
    gc.collect()
    return ctr

def gen_label_timedelta(time_slice, status_list, status):
    timedelta = [time_slice[i] for i,f in enumerate(status_list) if f in status]
    if len(timedelta)>0:
        timedelta = np.max(timedelta)
    else:
        timedelta = np.nan
    return timedelta

def gen_label_timedelta_min(time_slice, status_list, status):
    timedelta = [time_slice[i] for i,f in enumerate(status_list) if f in status]
    if len(timedelta)>0:
        timedelta = np.min(timedelta)
    else:
        timedelta = np.nan
    return timedelta

def gen_label_timedelta_diff(time_slice, status_list, status):
    timedelta = [time_slice[i] for i,f in enumerate(status_list) if f in status]
    if len(timedelta)>1:
        timedelta = np.mean(np.diff(timedelta))
    else:
        timedelta = np.nan
    return timedelta

def gen_timedelta_diff(df, d, status):
    df_ = df[(df['label'].isin(status))&(df['day']<d)].copy()
    df_ = df_[['link_id', 'label', 'future_slice_id', 'day']].drop_duplicates()
    df_ = df_.sort_values(by=['link_id', 'day', 'future_slice_id']).reset_index(drop=True)
    df_['shift_time'] = df_.groupby(['link_id', 'day'])['future_slice_id'].shift(1)
    print(df_.head())
    df_['timedelta'] = df_['future_slice_id'] - df_['shift_time']
    group_df = df_.groupby(['link_id'])['timedelta'].agg({'link_his_label_{}_timedelta_mean'.format('_'.join([str(f) for f in status])): 'mean',
                                                         'link_his_label_{}_timedelta_std'.format('_'.join([str(f) for f in status])): 'std',
                                                         'link_his_label_{}_timedelta_max'.format('_'.join([str(f) for f in status])): 'max',
                                                         'link_his_label_{}_timedelta_median'.format('_'.join([str(f) for f in status])): 'median',
                                                         'link_his_label_{}_timedelta_min'.format('_'.join([str(f) for f in status])): 'min'}).reset_index()
    group_df['day'] = d
    return group_df

def gen_over_speed_limit_features(speed, speed_limit):
    cnt = 0
    for s in speed:
        if s>=speed_limit*3.6:
            cnt+=1
    return cnt


def gen_status_speed_features(speed, status_list, status):
    speed = [speed[i] for i,f in enumerate(status_list) if f in status]
    if len(speed)>0:
        speed = np.mean(speed)
    else:
        speed = np.nan
    return speed

def get_ups():
    tmp_dict = defaultdict(list)
    with open('../raw_data/topo.txt') as f:
        for line in tqdm(f.readlines()):
            up, downs = line.split('\t')
            for d in downs.split(','):
                tmp_dict[d.strip()].append(up)
    tmp_df = pd.DataFrame([(k, v) for k, v in tmp_dict.items()], columns=['linkid', 'target_link_list'])
    tmp_df['linkid'] = tmp_df['linkid'].astype(int)
    tmp_df['target_link_list'] = tmp_df['target_link_list'].apply(lambda x: ','.join(x))
    return tmp_df


# 上下游节点的路况信息
def get_topo_info(df, topo_df, slices=30, mode='down'):
    if mode == 'down':
        flg = 'down_target_state'
    else:
        flg = 'up_target_state'
    use_ids = set(df['linkid'].unique())
    topo_df = topo_df[topo_df['linkid'].isin(use_ids)]
    topo_df['target_link_list'] = topo_df['target_link_list'].apply(lambda x: [int(c) for c in x.split(',') if int(c) in use_ids])
    topo_df['len'] = topo_df['target_link_list'].apply(len)
    topo_df = topo_df[topo_df['len'] > 0]
    del topo_df['len']
    curr_df = []
    for i in topo_df.values:
        for j in i[1]:
            curr_df.append([i[0], j])
    curr_df = pd.DataFrame(curr_df, columns=['linkid', 'target_id'])
    curr_df = curr_df.merge(df[['linkid', 'future_slice_id']], on='linkid', how='left')

    tmp_df = df[['linkid', 'current_slice_id', 'curr_state']]
    tmp_df.columns = ['target_id', 'current_slice_id', 'curr_state']

    curr_df = curr_df.merge(tmp_df, on='target_id', how='left')

    curr_df['{}_diff_slice'.format(flg)] = curr_df['future_slice_id'] - curr_df['current_slice_id']
    curr_df = curr_df[(curr_df['{}_diff_slice'.format(flg)] >= 0) & (curr_df['{}_diff_slice'.format(flg)] <= slices)]

    curr_df = curr_df.drop_duplicates()
    tmp_list = ['{}_diff_slice'.format(flg)]
    curr_df['{}_diff_slice'.format(flg)] = (slices - curr_df['{}_diff_slice'.format(flg)]) / slices
    for s in range(5):
        curr_df['{}_{}'.format(flg, s)] = curr_df['curr_state'].apply(lambda x: 1 if x == s else 0)
        curr_df['{}_{}'.format(flg, s)] = curr_df['{}_{}'.format(flg, s)] * curr_df['{}_diff_slice'.format(flg)]
        tmp_list.append('{}_{}'.format(flg, s))
    curr_df = curr_df.groupby(['linkid', 'future_slice_id'])[tmp_list].agg('sum').reset_index()
    return curr_df

def get_topo_info(df, topo_df, slices=30, mode='down'):
    if mode == 'down':
        flg = 'down_target_state'
    else:
        flg = 'up_target_state'
    use_ids = set(df['linkid'].unique())
    topo_df = topo_df[topo_df['linkid'].isin(use_ids)]
    topo_df['target_link_list'] = topo_df['target_link_list'].apply(lambda x: [int(c) for c in x.split(',') if int(c) in use_ids])
    topo_df['len'] = topo_df['target_link_list'].apply(len)
    topo_df = topo_df[topo_df['len'] > 0]
    del topo_df['len']
    curr_df = []
    for i in topo_df.values:
        for j in i[1]:
            curr_df.append([i[0], j])
    curr_df = pd.DataFrame(curr_df, columns=['linkid', 'target_id'])
    curr_df = curr_df.merge(df[['linkid', 'future_slice_id']], on='linkid', how='left')

    tmp_df = df[['linkid', 'current_slice_id', 'curr_state']]
    tmp_df.columns = ['target_id', 'current_slice_id', 'curr_state']

    curr_df = curr_df.merge(tmp_df, on='target_id', how='left')

    curr_df['{}_diff_slice'.format(flg)] = curr_df['future_slice_id'] - curr_df['current_slice_id']
    curr_df = curr_df[(curr_df['{}_diff_slice'.format(flg)] >= 0) & (curr_df['{}_diff_slice'.format(flg)] <= slices)]

    curr_df = curr_df.drop_duplicates()
    tmp_list = ['{}_diff_slice'.format(flg)]
    curr_df['{}_diff_slice'.format(flg)] = (slices - curr_df['{}_diff_slice'.format(flg)]) / slices
    for s in range(5):
        curr_df['{}_{}'.format(flg, s)] = curr_df['curr_state'].apply(lambda x: 1 if x == s else 0)
        curr_df['{}_{}'.format(flg, s)] = curr_df['{}_{}'.format(flg, s)] * curr_df['{}_diff_slice'.format(flg)]
        tmp_list.append('{}_{}'.format(flg, s))
    curr_df = curr_df.groupby(['linkid', 'future_slice_id'])[tmp_list].agg('sum').reset_index()
    return curr_df

class Optimizedkappa(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef*X_p
        report = classification_report(y, np.argmax(X_p, axis=1), digits=5, output_dict=True)
        score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6

        return -score

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(len(set(y)))]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Powell')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        report = classification_report(y, np.argmax(X_p, axis=1), digits=5, output_dict=True)
        score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6
        return score

    def coefficients(self):
        return self.coef_['x']


def f1_weight_score(preds, train_data):
    y_true = train_data.label
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    report = classification_report(y_true, preds, digits=5, output_dict=True)
#     print(report)
    score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6
    return 'f1_weight_score', score, True

def run_lgb(train, test, target, k):
    feats = [f for f in test.columns if f not in ['linkid', 'label', 'day', 'oof_pred_0',
                                                   'oof_pred_1', 'oof_pred_2', 'oof_pred', 'weekday',
                                                   'recent_speed_mean_vs_limit', 'key', 'gap', 'gap_abs',
                                                   'label_0_slice_diff', 'label_1_slice_diff', 'label_2_slice_diff',
                                                   'target_link_list']+drop_cols]
    print('Current num of features:', len(feats))
    folds = KFold(n_splits=k, shuffle=True, random_state=2020)
#     folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
    train_user_id = train['linkid'].unique()
    output_preds = []
    feature_importance_df = pd.DataFrame()
    offline_score = []
    train['oof_pred_0'] = 0
    train['oof_pred_1'] = 0
    train['oof_pred_2'] = 0
    for i, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):
        train_X, train_y = train.loc[train['linkid'].isin(train_user_id[train_idx]), feats], train.loc[train['linkid'].isin(train_user_id[train_idx]), target]
        test_X, test_y = train.loc[train['linkid'].isin(train_user_id[valid_idx]), feats], train.loc[train['linkid'].isin(train_user_id[valid_idx]), target]
        dtrain = lgb.Dataset(train_X,
                             label=train_y,
)
        dval = lgb.Dataset(test_X,
                           label=test_y,
                           )
        parameters = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'None',
            'num_leaves': 63,
            'num_class': 3,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'nthread': 12
        }
        lgb_model = lgb.train(
            parameters,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=f1_weight_score
        )
        train.loc[train['linkid'].isin(train_user_id[valid_idx]), 'oof_pred_0'] = lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration)[:, 0]
        train.loc[train['linkid'].isin(train_user_id[valid_idx]), 'oof_pred_1'] = lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration)[:, 1]
        train.loc[train['linkid'].isin(train_user_id[valid_idx]), 'oof_pred_2'] = lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration)[:, 2]
#         oof_probs[valid_idx] = lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration)
        op = Optimizedkappa()
        op.fit(train.loc[train['linkid'].isin(train_user_id[valid_idx]), ['oof_pred_0', 'oof_pred_1', 'oof_pred_2']].values, test_y)
        output_preds.append(op.coefficients()*lgb_model.predict(test[feats], num_iteration=lgb_model.best_iteration))

        valid_preds = np.argmax(train.loc[train['linkid'].isin(train_user_id[valid_idx]), ['oof_pred_0', 'oof_pred_1', 'oof_pred_2']].values* op.coefficients(), axis=1)
        report = classification_report(test_y, valid_preds, digits=5, output_dict=True)
        score = report['0.0']['f1-score'] * 0.2 + report['1.0']['f1-score'] * 0.2 + report['2.0']['f1-score'] * 0.6
        offline_score.append(score)

        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-F1 score:%.6f, OOF-STD:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(15))
    oof_preds = train[['oof_pred_0', 'oof_pred_1', 'oof_pred_2']].copy()
    del train['oof_pred_0'], train['oof_pred_1'], train['oof_pred_2']
    return output_preds, oof_preds, np.mean(offline_score)

if __name__ == '__main__':
    ### 数据准备 ###
    get_his_sample(20190705)
    train_path = '../user_data/sample_data.csv'
    test_path = '../raw_data/20190801_testdata.txt'

    tmp_train = day_origin_df(train_path)
    tmp_test = day_origin_df(test_path)
    supp_train = sample_train_data(tmp_train, tmp_test)

    train_df = pd.read_csv(train_path, header=None, sep=';')
    test = pd.read_table(test_path, sep=';', header=None)

    train_df = load_traffic_data(train_df)
    test['day'] = 20190801
    test = load_traffic_data(test)

    train_df = supp_train.merge(train_df, on=['linkid', 'future_slice_id', 'current_slice_id'], how='left')
    data = pd.concat([train_df, test], axis=0, ignore_index=True)

    topo = pd.read_csv('../raw_data/topo.txt', sep='\t', header=None, names=['linkid', 'target_link_list'])
    ups = get_ups()
    topo['down_cnt'] = topo['target_link_list'].apply(lambda x: x.count(',') + 1)
    ups['up_cnt'] = ups['target_link_list'].apply(lambda x: x.count(',') + 1)

    ### 特征工程 ###
    numeric_list = ['recent_speed', 'recent_eta', 'recent_vichles_num',
                    'history_speed_28', 'history_speed_21', 'history_speed_14', 'history_speed_7',
                    'history_eta_28', 'history_eta_21', 'history_eta_14', 'history_eta_7',
                    'history_vichles_num_28', 'history_vichles_num_21', 'history_vichles_num_14',
                    'history_vichles_num_7']
    for col in tqdm(numeric_list):
        data[col + '_mean'] = data[col].apply(lambda x: np.mean(x))
        data[col + '_max'] = data[col].apply(lambda x: np.max(x))
        data[col + '_min'] = data[col].apply(lambda x: np.min(x))
        data[col + '_std'] = data[col].apply(lambda x: np.std(x))
        data[col + '_median'] = data[col].apply(lambda x: np.median(x))

    status_list = ['recent_status', 'history_status_28', 'history_status_21', 'history_status_14', 'history_status_7']
    for col in tqdm(status_list):
        data[col + '_mean'] = data[col].apply(lambda x: np.mean(x))
        data[col + '_max'] = data[col].apply(lambda x: np.max(x))
        data[col + '_std'] = data[col].apply(lambda x: np.std(x))
        for i in range(0, 5):
            data[col + '_bef{}'.format(i + 1)] = data[col].apply(lambda x: x[i])
        for i in range(0, 5):
            data[col + '_{}_cnt'.format(i)] = data[col].apply(lambda x: x.count(i))

    file_names = ['20190701.txt',
                 '20190702.txt',
                 '20190703.txt',
                 '20190704.txt',
                 '20190705.txt',
                 '20190706.txt',
                 '20190707.txt',
                 '20190708.txt',
                 '20190709.txt',
                 '20190710.txt',
                 '20190711.txt',
                 '20190712.txt',
                 '20190713.txt',
                 '20190714.txt',
                 '20190715.txt',
                 '20190716.txt',
                 '20190717.txt',
                 '20190718.txt',
                 '20190719.txt',
                 '20190720.txt',
                 '20190721.txt',
                 '20190722.txt',
                 '20190723.txt',
                 '20190724.txt',
                 '20190725.txt',
                 '20190726.txt',
                 '20190727.txt',
                 '20190728.txt',
                 '20190729.txt',
                 '20190730.txt']

    his_df = []
    for path in tqdm(file_names):
        df = pd.read_table('../user_data/traffic/{}'.format(path), sep=';', header=None)
        df.columns = ['0', 'recent_feature', 'history_feature_28', 'history_feature_21', 'history_feature_14',
                      'history_feature_7']
        df['link_id'] = df['0'].apply(lambda x: int(x.split(' ')[0]))
        df['label'] = df['0'].apply(lambda x: int(x.split(' ')[1]))
        df['label'] = df['label'].apply(lambda x: 2 if x == 4 else x - 1)
        df['current_slice_id'] = df['0'].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df['0'].apply(lambda x: int(x.split(' ')[3]))
        df['day'] = path.split('.')[0]

        df['weekday'] = pd.to_datetime(df['day']).dt.weekday + 1
        df['hour'] = df['future_slice_id'].apply(lambda x: int(x / 30))
        df.drop(['0', 'recent_feature', 'history_feature_28', 'history_feature_21', 'history_feature_14',
                 'history_feature_7'], axis=1, inplace=True)
        his_df.append(df)
    his_df = pd.concat(his_df, axis=0, ignore_index=True)

    link_ctr = []
    for d in tqdm(file_names[2:] + ['20190801.txt']):
        link_ctr.append(gen_ctr_features(d.split('.')[0], ['link_id']))
    link_ctr = pd.concat(link_ctr, axis=0, ignore_index=True)

    future_slice_ctr = []
    for d in tqdm(file_names[2:] + ['20190801.txt']):
        future_slice_ctr.append(gen_ctr_features(d.split('.')[0], ['future_slice_id']))
    future_slice_ctr = pd.concat(future_slice_ctr, axis=0, ignore_index=True)

    link_hour_ctr = []
    for d in tqdm(file_names[2:] + ['20190801.txt']):
        link_hour_ctr.append(gen_ctr_features(d.split('.')[0], ['link_id', 'hour']))
    link_hour_ctr = pd.concat(link_hour_ctr, axis=0, ignore_index=True)

    link_hour_weekday_ctr = []
    for d in tqdm(file_names[2:] + ['20190801.txt']):
        link_hour_weekday_ctr.append(gen_ctr_features(d.split('.')[0], ['link_id', 'hour', 'weekday']))
    link_hour_weekday_ctr = pd.concat(link_hour_weekday_ctr, axis=0, ignore_index=True)

    del his_df
    gc.collect()

    link_ctr = link_ctr.rename(columns={'link_id': 'linkid'})
    link_ctr['day'] = link_ctr['day'].astype(int)
    future_slice_ctr['day'] = future_slice_ctr['day'].astype(int)
    link_hour_ctr = link_hour_ctr.rename(columns={'link_id': 'linkid'})
    link_hour_ctr['day'] = link_hour_ctr['day'].astype(int)
    link_hour_weekday_ctr = link_hour_weekday_ctr.rename(columns={'link_id': 'linkid'})
    link_hour_weekday_ctr['day'] = link_hour_weekday_ctr['day'].astype(int)

    data = data.merge(link_ctr, on=['linkid', 'day'], how='left')
    data = data.merge(future_slice_ctr, on=['future_slice_id', 'day'], how='left')
    data = data.merge(link_hour_ctr, on=['linkid', 'hour', 'day'], how='left')
    data = data.merge(link_hour_weekday_ctr, on=['linkid', 'hour', 'weekday', 'day'], how='left')

    data['label_0_max_slice'] = data['future_slice_id'] - data.apply(lambda x: gen_label_timedelta(x['recent_slices'], x['recent_status'], [1]), axis=1)
    data['label_1_max_slice'] = data['future_slice_id'] - data.apply(lambda x: gen_label_timedelta(x['recent_slices'], x['recent_status'], [2]), axis=1)
    data['label_2_max_slice'] = data['future_slice_id'] - data.apply(lambda x: gen_label_timedelta(x['recent_slices'], x['recent_status'], [3, 4]), axis=1)
    data['label_0_min_slice'] = data['future_slice_id'] - data.apply(lambda x: gen_label_timedelta_min(x['recent_slices'], x['recent_status'], [1]), axis=1)
    data['label_1_min_slice'] = data['future_slice_id'] - data.apply(lambda x: gen_label_timedelta_min(x['recent_slices'], x['recent_status'], [2]), axis=1)
    data['label_2_min_slice'] = data['future_slice_id'] - data.apply(lambda x: gen_label_timedelta_min(x['recent_slices'], x['recent_status'], [3, 4]), axis=1)
    data['label_0_slice_diff'] = data.apply(lambda x: gen_label_timedelta_diff(x['recent_slices'], x['recent_status'], [1]), axis=1)
    data['label_1_slice_diff'] = data.apply(lambda x: gen_label_timedelta_diff(x['recent_slices'], x['recent_status'], [2]), axis=1)
    data['label_2_slice_diff'] = data.apply(lambda x: gen_label_timedelta_diff(x['recent_slices'], x['recent_status'], [3, 4]), axis=1)

    ### 训练模型 ###
    train = data[data['day'] <= 20190730].copy()
    test = data[data['day'] == 20190801].copy().copy()
    train['label'] = train['label'] - 1
    drop_cols = numeric_list + status_list + ['recent_feature', 'history_feature_28', 'history_feature_21',
                                              'history_feature_14', 'history_feature_7', 'recent_slices']
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)
    print(train.shape, test.shape)

    down_df_trn = get_topo_info(train, topo, slices=15, mode='down')
    up_df_trn = get_topo_info(train, ups, slices=15, mode='up')
    down_df_tst = get_topo_info(test, topo, slices=15, mode='down')
    up_df_tst = get_topo_info(test, ups, slices=15, mode='up')

    train = train.merge(up_df_trn, on=['linkid', 'future_slice_id'], how='left')
    train = train.merge(down_df_trn, on=['linkid', 'future_slice_id'], how='left')

    test = test.merge(up_df_tst, on=['linkid', 'future_slice_id'], how='left')
    test = test.merge(down_df_tst, on=['linkid', 'future_slice_id'], how='left')

    lgb_preds, lgb_oof, lgb_score = run_lgb(train, test, 'label', 5)

    lgb_sub = test[['linkid', 'current_slice_id', 'future_slice_id']].copy()
    for i in range(0, 3):
        lgb_sub['lgb_pred_{}'.format(i)] = np.mean(lgb_preds, axis=0)[:, i]
    lgb_sub.to_csv('../user_data/lgb_sub_prob.csv', index=False)






