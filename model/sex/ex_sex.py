#import seaborn as sns
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from tiny.tfidf import *
from tiny.usage import *
from tiny.util import replace_invalid_filename_char, get_stable_feature


@timed()
def gen_sub_by_para(bal_ratio):
    args = locals()
    logger.debug(f'Run train dnn:{args}')

    drop_useless_pkg = True
    drop_long = 0.3
    n_topics = 5

    #feature_label = get_stable_feature('rf01')
    feature_label = get_stable_feature('1002')

    test = feature_label[feature_label['sex'].isnull()]
    train = feature_label[feature_label['sex'].notnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1, errors='ignore')
    Y = train['sex']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes, test_size=0.3, random_state=666)

    classifier = ExtraTreesClassifier(n_estimators=1000,
                                      max_depth=15,
                                      max_features=128,
                                      verbose=1,
                                      n_jobs=-1,
                                      random_state=42)


    print(f'Train begin#{args}')
    classifier.fit(X_train, y_train)
    print('Train End')


    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
    sub=pd.DataFrame(classifier.predict_proba(pre_x.values))


    sub.columns=Y_CAT.categories
    sub['DeviceID']=test['device'].values


    from sklearn.metrics import log_loss

    best = log_loss(y_test, classifier.predict_proba(X_test) )

    best_score = round(best, 4)

    #lgb.plot_importance(gbm, max_num_features=20)

    print(f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

    file = f'./sub/baseline_rf_ex_sex_{best}_{args}.csv'
    file = replace_invalid_filename_char(file)
    print(f'sub file save to {file}')
    sub = round(sub,10)
    sub.to_csv(file,index=False)

    print_imp_list(X_train, classifier)

    ###Save result for ensemble
    train_bk = pd.DataFrame(classifier.predict_proba(train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)),
                            index=train.device,
                            columns=Y_CAT.categories
                            )

    test_bk = pd.DataFrame(classifier.predict_proba(pre_x),
                           index=test.device,
                           columns=Y_CAT.categories
                           )

    save_result_for_ensemble(f'{best_score}__ex_sex_{args}',
                             train=train_bk,
                             test=test_bk,
                             label=None,
                             )

if __name__ == '__main__':

    gen_sub_by_para(0)
    # gen_sub_by_para(True, 0.4)
    # for drop_long in np.arange(0.1, 1.1, 0.1):
    #     for drop_useless_pkg in [True, False]:
    #
    #         gen_sub_by_para(drop_useless_pkg, round(drop_long,1))
