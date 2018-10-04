

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBClassifier

from tiny.lda import *
from  tiny.util import *


try:
    from tiny.conf import gpu_params
except :
    # GPU support
    gpu_params = {}




def gen_sub_by_para():
    args = locals()
    feature_label = get_stable_feature('1002')
    #feature_label = feature_label[['sex', 'phone_type', 'brand']]

    train = feature_label[feature_label['sex'].notnull()]

    train = balance_train(train, 0.5)
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1, errors='ignore')
    Y = train['sex']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes, test_size=0.3, random_state=666)


    gbm = XGBClassifier(
                    objective= 'binary:logistic', #''multi:softprob',
                    eval_metric= 'logloss',  #'mlogloss',
                    #num_class=22,
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
                    scale_pos_weight=0.95,

                    **gpu_params
                    )
    # print(random_search.grid_scores_)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=100, verbose=True )

    print_imp_list(X_train, gbm)

    results = gbm.evals_result()

    #print(results)

    best_epoch = np.array(results['validation_0']['logloss']).argmin() + 1
    best_score = np.array(results['validation_0']['logloss']).min()


    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
    sub=pd.DataFrame(gbm.predict_proba(pre_x))


    sub.columns=Y_CAT.categories
    sub['DeviceID']=test['device'].values

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')



    print(f'best_epoch:{best_epoch}_best_score:{best_score}')

    file = f'./sub/baseline_sex_{best_score}_{args}_epoch_{best_epoch}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub = round(sub,10)
    sub.to_csv(file,index=False)

    ###Save result for ensemble
    train_bk = pd.DataFrame(gbm.predict_proba(train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)),
                            index=train.device,
                            columns=Y_CAT.categories
                            )

    test_bk = pd.DataFrame(gbm.predict_proba(pre_x),
                           index=test.device,
                           columns=Y_CAT.categories
                           )


    save_result_for_ensemble(f'{best_score}_{best_epoch}_xgb_sex_0.95',
                             train=train_bk,
                             test=test_bk,
                             label=None,
                             )


if __name__ == '__main__':
   # for app_threshold in range(20, 3000, 10):
        gen_sub_by_para()
    #
    # par_list = list(np.round(np.arange(0, 0.01, 0.001), 5))
    # par_list.reverse()
    # print(par_list)
    # for learning_rate in par_list:
    #     #for colsample_bytree in np.arange(0.5, 0.8, 0.1):
    #         gen_sub_by_para(learning_rate)



