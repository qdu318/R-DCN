
import os
from DataPre import *


def csv_find(parent_path, file_flag):
    df_paths = []
    for root, dirs, files in os.walk(parent_path):

        for file in files:
            if "".join(file).find(file_flag) != -1:
                df_paths.append(root + '/' + file)
    return df_paths




def csv_find2(trn_for_pred,test_for_pred,args):
    csvs_TCN = []
    csvs_test_TCN = []
    for i in trn_for_pred['车号']:
        path = args.trn_path + '/' + str(i) + '.csv'
        csvs_TCN.append(path)
    for i in test_for_pred['车号']:
        path = args.test_path + '/' + str(i) + '.csv'
        csvs_test_TCN.append(path)
    return csvs_TCN,csvs_test_TCN

