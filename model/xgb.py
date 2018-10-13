

from xgboost import XGBClassifier

from tiny.lda import *
from  tiny.util import get_stable_feature, save_result_for_ensemble, summary_daily_usage

try:
    from tiny.conf import gpu_params
except :
    # GPU support
    gpu_params = {}




def gen_sub_by_para():

    args = locals()

    logger.debug(f'Run train dnn:{args}')
    #feature_label = get_dynamic_feature(None)
    feature_label = get_stable_feature('1011')

    #feature_label = random_feature(feature_label, 1/2)

    from tiny.feature_filter import get_best_feautre
    # feature_label = get_cut_feature(feature_label, drop_feature)

    #feature_label = get_best_feautre(feature_label)


    daily_info = summary_daily_usage()
    feature_label  = feature_label.merge(daily_info, on='device', how='left')

    train = feature_label[feature_label['sex'].notnull()]
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes)


    gbm = XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    #booster='dart',
                    num_class=22,
                    max_depth=3,
                    reg_alpha=10,
                    reg_lambda=10,
                    subsample=0.7,
                    colsample_bytree=0.6,
                    n_estimators=20000,


                    learning_rate=0.01,


                    seed=1,
                    missing=None,

                    #Useless Paras
                    silent=True,
                    gamma=0,
                    max_delta_step=0,
                    min_child_weight=1,
                    colsample_bylevel=1,
                    scale_pos_weight=1,

                    **gpu_params
                    )
    logger.debug(f"Run the xgb with:{gpu_params}")
    # print(random_search.grid_scores_)
    gbm.fit(X_train, y_train,  eval_set=[ (X_train, y_train), (X_test, y_test),], early_stopping_rounds=100, verbose=True )

    results = gbm.evals_result()

    #print(results)

    best_epoch = np.array(results['validation_1']['mlogloss']).argmin() + 1
    best_score = np.array(results['validation_1']['mlogloss']).min()


    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)

    print_imp_list(X_train, gbm)


    ###Save result for ensemble
    train_bk = pd.DataFrame(gbm.predict_proba(train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)),
                            index=train.device,
                            columns=Y_CAT.categories
                            )

    test_bk = pd.DataFrame(gbm.predict_proba(pre_x),
                           index=test.device,
                           columns=Y_CAT.categories
                           )

    label_bk = pd.DataFrame({'label': Y_CAT.codes},
                            index=train.device,
                            )

    save_result_for_ensemble(f'{best_score}_{best_epoch}_xgb_{len(feature_label.columns)}_{args}',
                             train=train_bk,
                             test=test_bk,
                             label=label_bk,
                             )

if __name__ == '__main__':
    # for svd_cmp in range(50, 200, 30):
    #for arg1 in range(99, 1000, 100):
        gen_sub_by_para()
    #
    # par_list = list(np.round(np.arange(0, 0.01, 0.001), 5))
    # par_list.reverse()
    # print(par_list)
    # for learning_rate in par_list:
    #     #for colsample_bytree in np.arange(0.5, 0.8, 0.1):
    #         gen_sub_by_para(learning_rate)



