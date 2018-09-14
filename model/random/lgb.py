from lightgbm.sklearn import LGBMRegressor, LGBMClassifier

from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from sklearn.cross_validation import train_test_split


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from tiny.lda import *
from  tiny.util import *

# New add
# deviceid_train.rename({'device_id':'device'}, axis=1, inplace=True)
deviceid_train = get_lda_from_app_install()
deviceid_train2 = get_lda_from_usage(mini=mini)

core_list = ['0', '1', '2', '3','4']
# for col in core_list:
#     deviceid_train_2[col] =  deviceid_train_2[col].apply(lambda val: 1 if val > 0 else 0)
deviceid_train = pd.concat([deviceid_train,deviceid_train2[core_list] ], axis=1)

deviceid_train = extend_feature(span_no=24, input=deviceid_train, trunc_long_time=False)

#deviceid_train = extend_feature(span_no=4,input=deviceid_train,  trunc_long_time=False)



#print(len(deviceid_train))
#deviceid_train.groupby('max_day_cnt')['max_day_cnt'].count()

#
#
# col_drop = [item for item in deviceid_train.columns if 'max_' in str(item)]
# deviceid_train.drop(columns=col_drop, inplace=True )

#deviceid_train.drop(columns=['tfidf_sum'], inplace=True )
deviceid_train.head()


train=deviceid_train[deviceid_train['sex'].notnull()]
test=deviceid_train[deviceid_train['sex'].isnull()]

X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
Y = train['sex_age']


gbm = LGBMClassifier(n_estimators=400,
                     boosting_type='gbdt',
                     objective='multiclass',
                     max_depth=-1,
                     num_class=22,
                     random_state=47,
                     metric='multi_logloss',
                     verbose=-1,
                     n_jobs=4,
                    )

#gbm.set_params(**params)

print(gbm)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
params  = {'num_leaves': sp_randint(6, 50),
             "min_data_in_leaf":sp_randint(500, 1500),
             'min_child_samples': sp_randint(100, 500),
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8),
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

random_search = RandomizedSearchCV(gbm, param_distributions=params,
                                   n_iter=param_comb, scoring='neg_log_loss', n_jobs=4,
                                   cv=skf.split(X,Y), verbose=3, random_state=1001 )

# print(random_search.grid_scores_)
random_search.fit(X, Y)
# print(random_search.grid_scores_)

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
