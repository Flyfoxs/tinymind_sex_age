#import seaborn as sns
import lightgbm as lgb
from sklearn.cross_validation import train_test_split

from tiny.lda import *
from  tiny.util import *

# New add
# deviceid_train.rename({'device_id':'device'}, axis=1, inplace=True)
deviceid_train = get_lda_feature()
deviceid_train = extend_feature(version='1',span_no=6, input=deviceid_train, trunc_long_time=900)

#print(len(deviceid_train))
#deviceid_train.groupby('max_day_cnt')['max_day_cnt'].count()
deviceid_train.head()




col_drop = [item for item in deviceid_train.columns if 'max_' in str(item)]
deviceid_train.drop(columns=col_drop, inplace=True )

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

    "min_data_in_leaf":1000,
    'verbose': -1

}



gbm = lgb.train(params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=lgb_eval,
        early_stopping_rounds=300)


best = round(gbm.best_score.get('valid_0').get('multi_logloss'), 5)
best


pre_x=test.drop(['sex','age','sex_age','device'],axis=1)
sub=pd.DataFrame(gbm.predict(pre_x.values,num_iteration=gbm.best_iteration))


sub.columns=Y_CAT.categories
sub['DeviceID']=test['device'].values
sub=sub[['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7','1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

file = f'./sub/baseline_{best}.csv'
print(f'sub file save to {file}')
sub.to_csv(file,index=False)


