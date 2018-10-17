import numpy as np
from code_felix.merge.utils import *
import numpy as np

from code_felix.merge.utils import *


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

    # #Best
    # file_list = [
    #     (0.25, 'd59044', './sub/ensemble_2.4139496899922688_epoch_110_drop_0.64_patience_50_lr_0.0005.csv'),
    #     (0.25, 'd5905', './sub/ensemble_2.423202401351929_epoch_168_drop_0.66_patience_50_lr_0.0005.csv'),
    #     (0.25, 'd1', './sub/_ensemble_2.5822630012512207_epoch_16_drop_0.63_dense_128_patience_50_lr_0.0005.csv'),
    #     (0.25, 'd2', './sub/_ensemble_2.5819819229125978_epoch_13_drop_0.6_dense_128_patience_50_lr_0.0005.csv'),
    #
    #
    #
    #     #(0.5, 'xg', './output/best/baseline_2.614742_2650_xgb_svd_cmp100.h5',),
    #     #(0.2, 'lg' , './sub/baseline_lg_sci_2.64374_learning_rate 0.02.csv'),
    #     #(0.5, 'd59488',  './sub/ensemble_2.44306288172404_epoch_79_drop_0.62_patience_50_lr_0.0005.csv'),
    #
    #     #(0.2, 'rfex', './sub/baseline_rf_ex_2.6577_label rf01, n_estimators 10000, max_depth 15.csv'),
    # ]
    for weight in np.arange(0.63, 0.68, 0.01):
        file_list = [
            #(weight, 'Fred', './output/best/neural_network_stacked_1test_include_v2.csv'),
            (weight, 'Fred', './sub/neural_network_stacked_team_v1_1test_huge.csv'),
            (round(1-weight, 2), 'd58664', './sub/merge_score_d59044_0.25_d5905_0.25_d1_0.25_d2_0.25.csv'),
            ]




        score = merge_score(file_list)
        score = round(score, 10)
        weight=[str(f'{file[1]}_{file[0]}') for file in file_list]
        file = f'./sub/merge_score_{"_".join(weight)}.csv'
        score.to_csv(file)
        print(file)



