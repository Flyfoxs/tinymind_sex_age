#import seaborn as sns
import lightgbm as lgb
from sklearn.cross_validation import train_test_split

from tiny.lda import *
from  tiny.util import *

# New add
# deviceid_train.rename({'device_id':'device'}, axis=1, inplace=True)
deviceid_train = get_lda_from_app_install(drop=False)
deviceid_train.set_index('device',inplace=True)

deviceid_train2 = get_lda_from_usage(mini=mini)
deviceid_train2.set_index('device',inplace=True)
#deviceid_train2.drop(columns=['device', 'sex', 'sex_age', 'age'], inplace=True)


#print(deviceid_train2.shape)

core_list = ['0', '1', '2', '3','4']
#print(f'========={deviceid_train3[core_list].columns}')
# for col in core_list:
#     deviceid_train_2[col] =  deviceid_train_2[col].apply(lambda val: 1 if val > 0 else 0)
deviceid_train = pd.concat([deviceid_train,deviceid_train2[core_list] ], axis=1)

deviceid_train = extend_feature(span_no=24, input=deviceid_train, trunc_long_time=False)

#deviceid_train = extend_feature(span_no=12,input=deviceid_train,  trunc_long_time=False)

drop_col = [col for col in deviceid_train.columns if col.endswith('_count') or col.endswith('_sum') ]
print(f'=========will drop:{drop_col}')
deviceid_train.drop(columns=drop_col, inplace=True)


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
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'max_depth': 3,
    'metric': {'multi_logloss'},
    'num_class': 22,
    'objective': 'multiclass',
    'random_state': 47,

    #"min_data_in_leaf":1000,
    'verbose': -1,
    'colsample_bytree': 0.58,
    # 'min_child_samples': 289,
    #'min_child_weight': 0.1,
    'min_data_in_leaf': 1472,
    #'num_leaves': 300,
    'reg_alpha': 3,
    'reg_lambda': 4,
    'subsample': 0.8


}




gbm = lgb.train(params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=lgb_eval,
        early_stopping_rounds=50)


best = round(gbm.best_score.get('valid_0').get('multi_logloss'), 5)
best


pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
sub=pd.DataFrame(gbm.predict(pre_x.values,num_iteration=gbm.best_iteration))


sub.columns=Y_CAT.categories
sub['DeviceID']=test['device'].values
sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]


#lgb.plot_importance(gbm, max_num_features=20)

print(f'=============Final train feature({len(deviceid_train.columns)}):\n{list(deviceid_train.columns)} \n {len(deviceid_train.columns)}')

file = f'./sub/baseline_{best}.csv'
print(f'sub file save to {file}')
sub.to_csv(file,index=False)

