import os
import argparse
from DataPre import *
from DataLoad import *
from Sample import *
from RuleClassfion import *
from model import TCN
from train_test_split import *
import pickle
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit



parser = argparse.ArgumentParser(description='TS')
parser.add_argument('--jobs', type=int, default=4)
parser.add_argument('--label_path', type=str, default='./datasets/data/train/train_labels.csv')
parser.add_argument('--trn_path',type=str,default='./datasets/data/train')
parser.add_argument('--test_path',type=str,default='./datasets/data/test')
parser.add_argument('--result_path',type=str,default='./work/prediction_result')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')


args = parser.parse_args()


if __name__=="__main__":


    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)


    trn_for_pred,test_for_pred=ruleClassfy(args)


    csvs_TCN,csvs_test_TCN=csv_find2(trn_for_pred,test_for_pred,args)
    df_trnall2_TCN, df_testall2_TCN= dataPre_TCN(csvs_TCN,csvs_test_TCN,args)


    df_trn=resample2(df_trnall2_TCN)

    cols = {'车号': 'Num', '整车钥匙状态catestd': 'key_std', }

    df_trn= df_trn.rename(columns=cols)
    print("df_trn",df_trn.shape)
    df_test = df_testall2_TCN.rename(columns=cols)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=2021)
    for trn_idx, val_idx in split.split(trn_for_pred['车号'], trn_for_pred['Label']):
        trn_data = df_trn[df_trn['Num'].apply(lambda x: x in trn_for_pred.iloc[trn_idx]['车号'].values)].reset_index(
            drop=True)
        val_data = df_trn[df_trn['Num'].apply(lambda x: x in trn_for_pred.iloc[val_idx]['车号'].values)].reset_index(
            drop=True)
    trn_data.to_csv("./train_fs.csv")
    val_data.to_csv("./test_fs.csv")









