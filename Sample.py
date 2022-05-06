
import pandas as pd
import numpy as np


def col_feature1(df):
    df = df [(df['电池包主负继电器状态cate']==0)]
    df = df [(df['if_off']<-2) | (df['车速']!=0)]
    return df


def resample(df_label,df_trnall2):
    df_label['CollectTime'] = pd.to_datetime(df_label['CollectTime'], format='%Y-%m-%d %H:%M:%S')

    df_label_new = df_label

    df_label1 = df_label[df_label['Label'] == 1]
    for kind, kind_df in df_label1.groupby('车号'):

        for t in np.arange(5):
            new_row1 = pd.DataFrame({'车号': kind, 'Label': 1,
                                     'CollectTime': kind_df['CollectTime'].iloc[0] + pd.Timedelta(seconds=t + 1)},
                                    index=[1])

            new_row2 = pd.DataFrame({'车号': kind, 'Label': 1,
                                     'CollectTime': kind_df['CollectTime'].iloc[0] - pd.Timedelta(seconds=t + 1)},
                                    index=[1])

            df_label_new = df_label_new.append(new_row1, ignore_index=True)
            df_label_new = df_label_new.append(new_row2, ignore_index=True)

    df = pd.merge(df_trnall2, df_label_new, on=['车号', 'CollectTime'], how='left')
    df['Label'] = df['Label'].fillna(0)
    df1 = df[df['Label'] == 1]
    return df,df_label_new


def col_feature2(df):
    cate_cols = ['制动踏板状态', '驾驶员离开提示', '主驾驶座占用状态', '驾驶员安全带状态',
                 '手刹状态', '整车钥匙状态', '整车当前档位状态']
    cate_cols2 = []

    for col in cate_cols:
        df[col + 'cate'] = df[col].astype('category').cat.codes
        cate_cols2.append(col + 'cate')
    df_code_dict = {col: {code: cate for code, cate in enumerate(df[col].astype('category').cat.categories)}
                    for col in cate_cols}
    print("value:",df_code_dict.values())
    df['整车钥匙状态catestd'] = df['整车钥匙状态cate'].rolling(window=5, center=True).std()


    df = df[(df['电池包主负继电器状态cate'] == 0)]
    df = df[(df['if_on'] == 0) | (df['车速'] > 20)]
    df = df[(df['if_off'] < -3) | (df['车速'] != 0)]


    bin1 = [-0.1, 0.01, 0.5, 1.5, 3, 100]
    df['v_bin'] = pd.cut(df['车速'], bin1, labels=False)

    df['a0'] = df.apply(lambda x: 1 if (x['加速踏板位置'] > 0) else 0, axis=1)
    df['a_min5'] = df['a_min5'] - df['v_bin'] * df['v_bin'] / 2 - df['a0'] * 1.5
    df['a_mean5'] = df['a_mean5'] - df['v_bin'] * df['v_bin'] / 2 - df['a0'] * 1.5
    df['v_diff1'] = df['v_diff1'] - df['v_bin'] * df['v_bin'] - df['a0'] * 3
    df['v_diff3'] = df['v_diff3'] - df['v_bin'] * df['v_bin'] - df['a0'] * 3

    bin1 = [-100, -15, -10, -6, -3, -0.01, 0.1, 100]
    df['v_diff3_bin'] = pd.cut(df['v_diff3'], bin1, labels=False)

    ori_cols = ['加速踏板位置', '电池包主负继电器状态', '电池包主正继电器状态', '制动踏板状态',
                '驾驶员离开提示', '主驾驶座占用状态', '驾驶员安全带状态', '手刹状态', '整车钥匙状态', '低压蓄电池电压',
                '整车当前档位状态', '整车当前总电流', '整车当前总电压', '车速', '方向盘转角']
    cate_cols = ['电池包主负继电器状态cate', '制动踏板状态cate', '驾驶员离开提示cate', '主驾驶座占用状态cate',
                 '驾驶员安全带状态cate', '手刹状态cate', '整车钥匙状态cate', '整车当前档位状态cate', ]
    choose_cols = ['time_delta', 'time_delta_5', 'if_on', 'a0', 'v_bin', 'v_diff3', 'v_diff2', 'v_diff4']
    df = df.drop(ori_cols + cate_cols + choose_cols, axis=1)

    return df, df_code_dict


def resample2(df_trnall2_TCN):
    df_label_new = pd.read_csv("./datasets/data/train/label_new.csv",index_col=0)
    df_label_new['CollectTime'] = pd.to_datetime(df_label_new['CollectTime'], format='%Y-%m-%d %H:%M:%S')
    df_TCN = pd.merge(df_trnall2_TCN, df_label_new, on=['车号', 'CollectTime'], how='left')
    df_TCN['Label'] = df_TCN['Label'].fillna(0)
    print('训练数据label==1数量:', df_TCN.shape)
    print(df_TCN.columns)
    print('训练数据label==1车数量:', df_TCN['车号'].nunique())
    return df_TCN