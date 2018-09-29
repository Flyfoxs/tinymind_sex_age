from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
import numpy as np
from xgboost import XGBClassifier

from tiny.util import *

try:
    from tiny.conf import gpu_params
except :
    # GPU support
    gpu_params = {}

feature_label = get_stable_feature('0924')



train=feature_label[feature_label['sex'].notnull()]
test =feature_label[feature_label['sex'].isnull()]

X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)[:10000]
Y = train['sex_age'][:10000]
Y_CAT = pd.Categorical(Y)
#X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)



lgbm =  LGBMClassifier(n_estimators=5000,
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

rf = ExtraTreesClassifier(n_estimators=1000,
                                      max_depth=15,
                                      max_features=128,
                                      verbose=1,
                                      n_jobs=-1,
                                      random_state=42)

xgb = XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    num_class=22,
                    max_depth=3,
                    reg_alpha=10,
                    reg_lambda=10,
                    subsample=0.7,
                    colsample_bytree=0.6,
                    n_estimators=2000,


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


lr = LogisticRegression()
sclf = StackingCVClassifier(classifiers=[lgbm, xgb, rf],
                          #use_probas=True,
                         # average_probas=False,
                          meta_classifier=lr)




X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.codes, test_size=0.3, random_state=666)

logger.debug("Fix begin")
sclf.fit(X_train.values, y_train)
logger.debug("Fix End")

# #train = feature_label[feature_label['sex'].notnull()]
# test = feature_label[feature_label['sex'].isnull()]

from sklearn.metrics import log_loss

best = log_loss(y_test, sclf.predict_proba(X_test.values))

# lgb.plot_importance(gbm, max_num_features=20)

print(
    f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
sub = pd.DataFrame(sclf.predict_proba(pre_x.values))

sub.columns = Y_CAT.categories
sub['DeviceID'] = test['device'].values
sub = sub[
    ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
     '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

file = f'./sub/baseline_st_{best}.csv'
file = replace_invalid_filename_char(file)
print(f'sub file save to {file}')
sub.to_csv(file, index=False)