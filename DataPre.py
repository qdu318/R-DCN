import os
import pandas as pd
import numpy as np
from joblib import Parallel,delayed
from tqdm.notebook import tqdm
from Sample import *



def label_clean(label_path):
    df_label = pd.read_csv(label_path)
    df_label.loc[4, :] = (5, 0, np.NaN)
    df_label.loc[18, :] = (19, 0, np.NaN)
    df_label.loc[35, :] = (36, 0, np.NaN)
    df_label.loc[93, :] = (94, 0, np.NaN)
    df_label.loc[76, :] = (77, 1, '2020/10/20 13:37:12')
    print(df_label.head())
    return df_label

def read_csv(path):
    del_cols = ['车辆行驶里程', '驾驶员需求扭矩值']
    num_cols = ['低压蓄电池电压', '整车当前总电流', '整车当前总电压']


    df0 = pd.read_csv(path, index_col=False, low_memory=False)
    df = df0.dropna(axis=0, thresh=17)
    df = df.rename(columns={'采集时间': 'CollectTime'})
    df['CollectTime'] = pd.to_datetime(df['CollectTime'], format='%Y-%m-%d %H:%M:%S')
    df.drop_duplicates(subset=['车号', 'CollectTime'], keep='first', inplace=True)
    df = df.sort_values('CollectTime').reset_index(drop=True)

    df['整车当前档位状态'] = df['整车当前档位状态'].replace('驻车', '空档')

    df['电池包主负继电器状态'] = df['电池包主负继电器状态'].replace('粘连', '断开')
    df = df[df['主驾驶座占用状态'] != '传感器故障']


    df['time_delta'] = df['CollectTime'].diff().dt.total_seconds()
    df['time_delta_5'] = (df['CollectTime'] - df['CollectTime'].shift(5)).dt.total_seconds()

    a = pd.DataFrame()
    b = pd.DataFrame()

    df['电池包主负继电器状态cate'] = df['电池包主负继电器状态'].astype('category').cat.codes
    for i in np.arange(5):

        a['电池状态' + str(i)] = df['电池包主负继电器状态cate'] - df['电池包主负继电器状态cate'].shift(i + 1)

        b['电池状态' + str(i)] = df['电池包主负继电器状态cate'] - df['电池包主负继电器状态cate'].shift(-i - 1)

    df['if_off'] = a.sum(axis=1)
    df['if_on'] = b.sum(axis=1)


    df['v_diff1'] = df['车速'].diff() / df['time_delta']
    df['v_diff2'] = -df['车速'].shift(3).rolling(window=3).mean() / df['time_delta_5']
    df['v_diff3'] = df['车速'].diff()
    df['v_diff4'] = df['v_diff1'].shift(-1)
    df['a_min5'] = df['v_diff1'].rolling(window=3).min()
    df['a_mean5'] = df['v_diff1'].rolling(window=3).mean()
    df['a_max3'] = df['v_diff1'].rolling(window=3).max()

    df = df.iloc[5:, :]


    df = df[df['time_delta_5'] < 90]

    return df

def applyParallel_concat(paths, func, jobs=4):

    ret = Parallel(n_jobs=jobs)(delayed(func)(csv)for csv in tqdm(paths))
    return pd.concat(ret)




def dataPre_TCN(csvs_TCN,csvs_test_TCN,args):

    df_trnall_TCN = applyParallel_concat(csvs_TCN, read_csv, jobs=args.jobs)
    df_testall_TCN = applyParallel_concat(csvs_test_TCN, read_csv, jobs=args.jobs)

    df_trnall2_TCN, df_code_dict1 = col_feature2(df_trnall_TCN)

    df_testall2_TCN, df_code_dict2 = col_feature2(df_testall_TCN)


    return df_trnall2_TCN, df_testall2_TCN

