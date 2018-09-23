from keras import models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from tiny.tfidf import *
from tiny.usage import *

tmp_model = './model/checkpoint/dnn_best_tmp.hdf5'

def train_dnn(dense1, dense2):

    dropout = 0.6
    args = locals()
    feature_label = get_stable_feature('0922')

    train = feature_label[feature_label['sex'].notnull()]
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)

    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)


    print(X_train.shape, y_train.shape)
    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(dense1, input_shape=(input_dim,)))
    #model.add(Activation('sigmoid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(dropout))


    model.add(Dense(dense2, ))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())


    model.add(Dense(15, ))
    model.add(LeakyReLU(alpha=0.01))


    model.add(Dense(22, ))
    model.add(Activation('softmax'))

    # model.compile(optimizer="sgd", loss="mse")
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                    #metrics=['categorical_crossentropy'],
                  )
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

    #'./model/'
    file_path = tmp_model

    check_best = ModelCheckpoint(filepath=replace_invalid_filename_char(file_path),
                                monitor='val_loss',verbose=1,
                                save_best_only=True, mode='min')

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=50,
                               )

    history = model.fit(X_train, np_utils.to_categorical(y_train),
                        validation_data=(X_test, np_utils.to_categorical(y_test)),
                        callbacks=[check_best, early_stop],
                        batch_size=128,
                        #steps_per_epoch= len(X_test)//128,
                        epochs=500, verbose=1)

    return model, history, args


if __name__ == '__main__':
    #for drop in np.arange(0.4, 0.8, 0.05):
        for dense1 in np.arange(800, 1500, 100):
            for dense2 in np.arange(80, 150, 10):

                _ , history, args = train_dnn(dense1, dense2)

            model = models.load_model(tmp_model)

            feature_label = get_stable_feature('0922')

            train = feature_label[feature_label['sex'].notnull()]
            test = feature_label[feature_label['sex'].isnull()]

            X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)

            Y = train['sex_age']
            Y_CAT = pd.Categorical(Y)
            X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)


            classifier = model

            pre_x = test.drop(['sex', 'age', 'sex_age', 'device'], axis=1)
            sub = pd.DataFrame(classifier.predict_proba(pre_x.values))

            sub.columns = Y_CAT.categories
            sub['DeviceID'] = test['device'].values
            sub = sub[
                ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
                 '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]

            from sklearn.metrics import log_loss

            best = log_loss(y_test, classifier.predict_proba(X_test))

            model_file = f'./model/checkpoint/dnn_best_{best}_{args}.hdf5'
            model.save(model_file,
                       overwrite=True)

            print(
                f'=============Final train feature({len(feature_label.columns)}):\n{list(feature_label.columns)} \n {len(feature_label.columns)}')

            file = f'./sub/baseline_dnn_{best}_{args}.csv'
            file = replace_invalid_filename_char(file)
            print(f'sub file save to {file}')
            sub = round(sub, 10)
            sub.to_csv(file, index=False)