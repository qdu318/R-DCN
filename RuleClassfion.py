
import os
import argparse
from DataPre import *
from DataLoad import *
from Sample import *

def ruleClassfy(args):

    df_label = label_clean(args.label_path)


    csvs = csv_find(args.trn_path, file_flag='csv')
    csvs.remove(args.label_path)
    csvs_test = csv_find(args.test_path, file_flag="csv")

    df_trnall = applyParallel_concat(csvs, read_csv, jobs=args.jobs)



    df_testall = applyParallel_concat(csvs_test, read_csv, jobs=args.jobs)

    df_trnall2 = col_feature1(df_trnall)
    df_testall2 = col_feature1(df_testall)


    df,df_label_new = resample(df_label, df_trnall2)
    df_label_new.to_csv("./datasets/data/train/label_new.csv")


    feat = (df['v_diff1'] < -9) | (df['v_diff2'] < -6) | (df['v_diff3'] < -40)
    feat_test = (df_testall2['v_diff1'] < -9) | (df_testall2['v_diff2'] < -6) | (df_testall2['v_diff3'] < -40)

    df_trn_label = df[feat]
    sub_trn = pd.DataFrame(columns=['车号', 'Label_pred', 'CollectTime_pred'])

    sub_trn = pd.merge(df_label, sub_trn, on=['车号'], how='left')
    for kind, kind_df in df_trn_label.groupby('车号'):

        sub_trn.loc[sub_trn[sub_trn['车号'] == kind].index, 'Label_pred'] = 1

        sub_trn.loc[sub_trn[sub_trn['车号'] == kind].index, 'CollectTime_pred'] = kind_df['CollectTime'].iloc[0]

    sub_trn['CollectTime_pred'] = pd.to_datetime(sub_trn['CollectTime_pred'], format='%Y-%m-%d %H:%M:%S')
    sub_trn.to_csv("./datasets/data/rule_trn.csv")
    print('trn预测最大时间差:', np.abs(sub_trn['CollectTime_pred'] - sub_trn['CollectTime']).max())
    cols = ['车号', 'Label', 'CollectTime']
    trn_for_pred = sub_trn[sub_trn['Label_pred'] != 1][cols]
    print('trn需预测数据:', trn_for_pred.shape)

    sub_rule = pd.DataFrame(columns=['车号', 'Label', 'CollectTime'])
    sub_rule['车号'] = np.arange(121, 261)
    for kind, kind_df in df_testall2[feat_test].groupby('车号'):
        sub_rule.loc[sub_rule[sub_rule['车号'] == kind].index, 'Label'] = 1
        sub_rule.loc[sub_rule[sub_rule['车号'] == kind].index, 'CollectTime'] = kind_df['CollectTime'].iloc[0]
    test_for_pred = sub_rule[sub_rule['Label'] != 1][cols]
    trn_for_pred.to_csv('./train_label_for_pred.csv',index=False)
    test_for_pred.to_csv('./test_label_for_pred.csv',index=False)
    sub_rule.to_csv('./submit_rule.csv',index=False)

    return trn_for_pred,test_for_pred

