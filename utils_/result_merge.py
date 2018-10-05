from keras import Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dropout, Dense, LeakyReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from tiny.util import attach_device_train_label, replace_invalid_filename_char
from utils_.util_log import *

def read_result_for_ensemble(file):
    #file = f'./output/best/{name}.h5'
    store = pd.HDFStore(file)
    ensemble = (store["train"],
                store["label"] if 'label' in store else None,
                store["test"])
    store.close()
    return ensemble

def get_label_cat():
    label =  attach_device_train_label(None)
    return pd.Categorical(label.sex_age).categories



file_list = [
    #'./output/best/2.621213_2510_xgb.h5' ,
    './output/best/baseline_2.614742_2650_xgb_svd_cmp100.h5' ,
    './output/best/baseline_2.62099_287_lgb_min_data_in_leaf1472.h5' ,

    './output/best/2.635281090037028_1569_dnn.h5' ,

    #Sex
    './output/best/0.608252_2577_xgb_sex.h5' ,
    './output/best/0.625989340877533_357_v_1002_dnn.h5' ,

    #Age
    './output/best/baseline_2.004356_3384_xgb_age_svd_cmp50.h5' ,


]

train_list =[]
label_list = []
test_list  = []
for file in file_list:
    train, label, test = read_result_for_ensemble(file)

    train_list.append(train)
    if label is not None: label_list.append(label)
    test_list.append(test)

train = pd.concat(train_list, axis=1)
test = pd.concat(test_list, axis=1)
label = label_list[0]


train = train.sort_index()
label = label.sort_index()

X_train, X_test, y_train, y_test = train_test_split(train, label.iloc[:,0], test_size=0.3, random_state=234)

drop_out = 0.4
patience=50
lr = 0.0005
#搭建融合后的模型
inputs = Input((X_train.shape[1:]))

x = Dropout(drop_out)(inputs)

x = Dense(128, activation='relu')(x)

x = Dropout(drop_out)(x)

x = Dense(22, activation='softmax')(x)
model = Model(inputs, x)


########################################

# np.random.seed(1337)
#
# import tensorflow as tf
# tf.set_random_seed(1234)
#
# import random as rn
# rn.seed(12345)

early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                           patience=patience,
                           )

model_file ='./model/checkpoint/ensemble.h5'
check_best = ModelCheckpoint(filepath= model_file,
                             monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

reduce = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=patience//2,verbose=1,mode='min')

from keras.utils import np_utils
adam = Adam(lr)
model.compile(loss='categorical_crossentropy', optimizer=adam,
              # loss="binary_crossentropy", optimizer="adam",
              # metrics=["accuracy"]
              )


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print(np_utils.to_categorical(y_train).shape)

history = model.fit(X_train, np_utils.to_categorical(y_train),
                    validation_data=(X_test, np_utils.to_categorical(y_test)),
                    callbacks=[check_best,
                               early_stop,
                               reduce,
                               ],
                    batch_size=128,
                    # steps_per_epoch= len(X_test)//128,
                    epochs=10000,
                    verbose=1,

                    )

from keras import models
model_load = models.load_model(model_file)

best_epoch = np.array(history.history['val_loss']).argmin() + 1
best_score = np.array(history.history['val_loss']).min()

#pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
sub = pd.DataFrame(model_load.predict(test), columns=get_label_cat())


sub['DeviceID'] = test.index.values
sub = sub[
    ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
     '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

file = f'./sub/ensemble_{best_score}_epoch_{best_epoch}_drop_{drop_out}_patience_{patience}_lr_{lr}.csv'
file = replace_invalid_filename_char(file)
logger.debug(f'Input dim is {train.shape}')
logger.info(f'sub file save to {file}')
sub = round(sub, 10)
sub.to_csv(file, index=False)


