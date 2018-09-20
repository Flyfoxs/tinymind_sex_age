from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from sklearn.cross_validation import train_test_split

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
Y_CAT = pd.Categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)
# lgb_train = lgb.Dataset(X_train, label=y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    #'boosting_type': 'gbdt',
    'max_depth': 3,
    #'metric': {'multi_logloss'},
    #'num_class': 22,
    #'objective': 'multiclass',
    #'random_state': 47,
    "min_data_in_leaf":1000,
    'verbose': -1
}

params = {
        'boosting_type': 'gbdt',
        'max_depth': 3,
        'metric': {'multi_logloss'},
        'num_class': 22,
        'objective': 'multiclass',
        'random_state': 47,
        'verbose': -1,
        'feature_fraction': 0.2,
        'min_data_in_leaf': 1472,

        'reg_alpha': 3,
        'reg_lambda': 4,
        'subsample': 0.2

    }


gbm = LGBMClassifier(n_estimators=1000,
                     boosting_type='gbdt',
                     objective='multiclass',
                     num_class=22,
                     random_state=47,
                     eval_metric='multi_logloss',


                     subsample=0.2,
                     feature_fraction=0.2,
                     reg_alpha=3,
                     reg_lambda=4,
                     min_data_in_leaf=1472,
                     max_depth= 3,
                    )

gbm.set_params(**params)

print(gbm)

gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)],early_stopping_rounds=100,)

#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)#
print('Feature importances:', list(gbm.feature_importances_))

best = round(gbm.best_score_.get('valid_0').get('multi_logloss'), 5)
best


pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
sub=pd.DataFrame(gbm.predict_proba(pre_x.values,num_iteration=gbm.best_iteration_))


sub.columns=Y_CAT.categories
sub['DeviceID']=test['device'].values
sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

file = f'./sub/baseline_{best}.csv'
print(f'sub file save to {file}')
sub.to_csv(file,index=False)

#lgb.plot_importance(gbm, max_num_features=20)

print(f'=============\nFinal train feature:{list(deviceid_train.columns)}')


