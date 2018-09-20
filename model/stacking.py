from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
import numpy as np
from tiny.util import *

feature_label = get_stable_feature()



train=feature_label[feature_label['sex'].notnull()]
test =feature_label[feature_label['sex'].isnull()]

X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)



Y = train['sex_age']
Y_CAT = pd.Categorical(Y)
#X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)



lgbm =  LGBMClassifier(n_estimators=1000,
                     boosting_type='gbdt',
                     objective='multiclass',
                     num_class=22,
                     random_state=47,
                     eval_metric='logloss',
                     subsample=0.5,
                     feature_fraction=0.2,
                     reg_alpha=3,
                     reg_lambda=4,
                     min_data_in_leaf=1472,
                     max_depth= 3,
                     verbose=-1,
                    )
rf = RandomForestClassifier(n_estimators=5000,
                                        #criterion='entropy',
                                        verbose=1,
                                        n_jobs=-1,
                                        random_state=42)

lr = LogisticRegression()
sclf = StackingCVClassifier(classifiers=[lgbm, rf],
                          #use_probas=True,
                         # average_probas=False,
                          meta_classifier=lr)

#sclf = lgbm




X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)

sclf.fit(X_train.values, y_train)


# #train = feature_label[feature_label['sex'].notnull()]
# test = feature_label[feature_label['sex'].isnull()]

from sklearn.metrics import log_loss

best = log_loss(y_test, sclf.predict_proba(X_test))

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