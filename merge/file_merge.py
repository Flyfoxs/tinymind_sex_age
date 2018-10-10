
import pandas as pd
from merge.dnn_merge import *
def merge_score(file_list):
    df_merge = None
    for  weight, name, file  in file_list:
        if file.endswith('.h5'):
            _, _, df = read_result_for_ensemble(file)
        else:
            df = pd.read_csv(file, index_col ='DeviceID')

        df = df * weight

        if df_merge is None:
           df_merge = df
        else:
           df_merge = df_merge+df

    df_merge = df_merge[
        ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
         '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
    #
    return df_merge

if __name__ == '__main__':

    #Best
    file_list = [
        (0.5, 'd59453', './sub/ensemble_2.4256239612579344_epoch_55_drop_0.59_patience_50_lr_0.0005.csv'),
        (0.5, 'xg', './output/best/baseline_2.614742_2650_xgb_svd_cmp100.h5',),
        #(0.2, 'lg' , './sub/baseline_lg_sci_2.64374_learning_rate 0.02.csv'),
        #(0.5, 'd59488',  './sub/ensemble_2.44306288172404_epoch_79_drop_0.62_patience_50_lr_0.0005.csv'),

        #(0.2, 'rfex', './sub/baseline_rf_ex_2.6577_label rf01, n_estimators 10000, max_depth 15.csv'),
    ]



    score = merge_score(file_list)
    score = round(score, 10)
    weight=[str(f'{file[1]}_{file[0]}') for file in file_list]
    file = f'./sub/merge_score_{"_".join(weight)}.csv'
    score.to_csv(file)
    print(file)



