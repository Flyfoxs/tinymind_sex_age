

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
    feature_label = get_dynamic_feature()

    train = feature_label[feature_label['sex'].notnull()]
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes, test_size=0.3, random_state=666)


    gbm = XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
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
    # print(random_search.grid_scores_)
    gbm.fit(X_train, y_train,  eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=True )

    results = gbm.evals_result()

    #print(results)

    best_epoch = np.array(results['validation_0']['mlogloss']).argmin() + 1
    best_score = np.array(results['validation_0']['mlogloss']).min()


    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
    sub=pd.DataFrame(gbm.predict_proba(pre_x))


    sub.columns=Y_CAT.categories
    sub['DeviceID']=test['device'].values
    sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]


    from sklearn.metrics import log_loss

    best = log_loss(y_test, gbm.predict_proba(X_test) )

    best = round(best, 4)

    #lgb.plot_importance(gbm, max_num_features=20)

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')



    print(f'best_epoch:{best_epoch}_best_score:{best_score}')

    file = f'./sub/baseline_xgb_{best}_{args}_epoch_{best_epoch}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub = round(sub,10)
    sub.to_csv(file,index=False)

    print_imp_list(X_train, gbm)

if __name__ == '__main__':
    gen_sub_by_para()
    #
    # par_list = list(np.round(np.arange(0, 0.01, 0.001), 5))
    # par_list.reverse()
    # print(par_list)
    # for learning_rate in par_list:
    #     #for colsample_bytree in np.arange(0.5, 0.8, 0.1):
    #         gen_sub_by_para(learning_rate)



