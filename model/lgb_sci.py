#import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
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
    #train = balance_train(train)
    test =feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)

    gbm = LGBMClassifier(n_estimators=2000,
                         boosting_type='gbdt',
                         objective='multiclass',
                         num_class=22,
                         random_state=47,
                         metric=['multi_logloss'],
                         verbose=-1,
                         max_depth=3,


                         feature_fraction=0.2,
                         subsample=0.5,
                         min_data_in_leaf=1472,

                         reg_alpha=2,
                         reg_lambda=4,

                         ##########
                         learning_rate=0.02,  # 0.1
                         colsample_bytree=None,  #1
                         min_child_samples=None,  #20
                         min_child_weight=None,  #0.001
                         min_split_gain=None,  #0
                         num_leaves=None,  #31
                         subsample_for_bin=None,  #200000
                         subsample_freq=None,  #1
                         nthread=-1,

                         )

    # gbm.set_params(**params)

    print(gbm)

    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, )

    print('Feature importances:', list(gbm.feature_importances_))

    print_imp_list(X_train, gbm)

    best = round(gbm.best_score_.get('valid_0').get('multi_logloss'), 5)
    best

    best = "{:.5f}".format(best)

    pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    sub = pd.DataFrame(gbm.predict_proba(pre_x.values, num_iteration=gbm.best_iteration_))

    sub.columns=Y_CAT.categories
    sub['DeviceID']=test['device'].values
    sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

    # from sklearn.metrics import log_loss
    # loss = log_loss(y_test, gbm.predict_proba(X_test,num_iteration=gbm.best_iteration_))
    #
    # print(f'Loss={loss}, best={best}')
    #lgb.plot_importance(gbm, max_num_features=20)

    #print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_lg_sci_{best}_{args}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub.to_csv(file,index=False)

if __name__ == '__main__':
    #for learning_rate in np.arange(0.02, 0.02, 0.01):
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
