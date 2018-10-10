

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBClassifier

from tiny.lda import *
from tiny.util import *

import sys

sys.path.insert(0, "./")

from gcforest.gcforest import GCForest

try:
    from tiny.conf import gpu_params
except :
    # GPU support
    gpu_params = {}



def get_sex_age_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 22
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def gen_sub_by_para(svd_cmp):
    #version = '1002'
    args = locals()
    logger.debug(f'Run train dnn:{args}')
    #feature_label = get_dynamic_feature(svd_cmp)
    feature_label = get_stable_feature('1006')

    feature_label =  feature_label[:10000]
    #feature_label = random_feature(feature_label, 2/3)

    train = feature_label[feature_label['sex'].notnull()]
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes, test_size=0.3, random_state=666)


    gc = GCForest(get_sex_age_config())  # should be a dict
    X_train_enc = gc.fit_transform(X_train.values, y_train, X_test.values, y_test)
    y_pred = gc.predict(X_test)

    from sklearn.metrics import log_loss
    loss = log_loss(y_test, gc.predict_proba(X_test))

    logger.debug("the result in validate is :%s" % loss)

    pre_x=test.drop(['sex','age','sex_age','device'],axis=1)

    print_imp_list(X_train, gc)


    ###Save result for ensemble
    train_bk = pd.DataFrame(gc.predict_proba(train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)),
                            index=train.device,
                            columns=Y_CAT.categories
                            )

    test_bk = pd.DataFrame(gc.predict_proba(pre_x),
                           index=test.device,
                           columns=Y_CAT.categories
                           )

    label_bk = pd.DataFrame({'label': Y_CAT.codes},
                            index=train.device,
                            )

    save_result_for_ensemble(f'{best_score}_gc_{len(feature_label.columns)}_{args}',
                             train=train_bk,
                             test=test_bk,
                             label=label_bk,
                             )

if __name__ == '__main__':
    # for svd_cmp in range(50, 200, 30):
        gen_sub_by_para(0)
    #
    # par_list = list(np.round(np.arange(0, 0.01, 0.001), 5))
    # par_list.reverse()
    # print(par_list)
    # for learning_rate in par_list:
    #     #for colsample_bytree in np.arange(0.5, 0.8, 0.1):
    #         gen_sub_by_para(learning_rate)



