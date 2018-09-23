#import seaborn as sns
import lightgbm as lgb
from sklearn.cross_validation import train_test_split

from tiny.tfidf import *
from tiny.usage import *


@timed()
def gen_sub_by_para():
    args = locals()

    drop_useless_pkg=True
    drop_long =0.3
    n_topics=5

    lda_feature = get_lda_from_usage(n_topics)

    feature = extend_feature(span_no=24, input=lda_feature,
                             drop_useless_pkg=drop_useless_pkg, drop_long=drop_long)

    feature = convert_label_encode(feature)


    feature_label = attach_device_train_label(feature)



    train=feature_label[feature_label['sex'].notnull()]
    test =feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',

        'metric': {'multi_logloss'},
        'num_class': 22,
        'objective': 'multiclass',
        'random_state': 47,
        'verbose': -1,
        'max_depth': 3,
        #"min_data_in_leaf":1000,

        'feature_fraction': 0.2,
        'subsample': 0.5,
        # 'min_child_samples': 289,
        #'min_child_weight': 0.1,
        'min_data_in_leaf': 1472,
        #'num_leaves': 300,
        'reg_alpha': 2,
        'reg_lambda': 4,


    }


    try:

        gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,)

        print(f"Light GBM:{gbm.}")
    except Exception as error:
        print(f'Model input columns:{list(X.columns)}\n dict({X.dtypes.sort_values()})')
        raise error


    best = round(gbm.best_score.get('valid_0').get('multi_logloss'), 5)

    best = "{:.5f}".format(best)

    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
    sub=pd.DataFrame(gbm.predict(pre_x.values,num_iteration=gbm.best_iteration))


    sub.columns=Y_CAT.categories
    sub['DeviceID']=test['device'].values
    sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
    from sklearn.metrics import log_loss
    loss = log_loss(y_test, gbm.predict(X_test,num_iteration=gbm.best_iteration))

    print(f'Loss={loss}')
    #lgb.plot_importance(gbm, max_num_features=20)

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_lg_{best}_{args}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub.to_csv(file,index=False)

if __name__ == '__main__':
    # for reg_alpha in np.arange(1, 5, 1):
    #     for reg_lambda in np.arange(1, 5, 1):
            gen_sub_by_para()
    # #for limit in range(100, 1300, 100):
    # for drop in np.arange(0.1, 1.1, 0.1):
    #     gen_sub_by_para(True, round(drop, 2), n_topics=5)
    # gen_sub_by_para(True, 0.4)
    # for drop_long in np.arange(0.1, 1.1, 0.1):
    #     for drop_useless_pkg in [True, False]:
    #
    #         gen_sub_by_para(drop_useless_pkg, round(drop_long,1))
