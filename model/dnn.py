from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from model.checkpoint import ReviewCheckpoint
from tiny.usage import *


def train_dnn(dropout, dense, epochs):
    args = locals()
    feature_label = get_stable_feature()

    train = feature_label[feature_label['sex'].notnull()]
    test = feature_label[feature_label['sex'].isnull()]

    X = train.drop(['sex', 'age', 'sex_age', 'device'], axis=1)

    Y = train['sex_age']
    Y_CAT = pd.Categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_CAT.labels, test_size=0.3, random_state=666)


    print(X_train.shape, y_train.shape)
    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(Dense(1000, input_shape=(input_dim,)))
    #model.add(Activation('sigmoid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(100, activation=LeakyReLU(alpha=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(100, activation=LeakyReLU(alpha=0.01)))
    model.add(BatchNormalization())


    model.add(Dense(dense, activation=LeakyReLU(alpha=0.01)))


    model.add(Dense(22, activation=Activation('softmax')))


    # model.compile(optimizer="sgd", loss="mse")
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                    #metrics=['categorical_crossentropy'],
                  )
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

    review = ReviewCheckpoint(X_test, y_test, args)

    history = model.fit(X_train, np_utils.to_categorical(y_train),
                        validation_data=(X_test, np_utils.to_categorical(y_test)),
                      callbacks=[review],
                      epochs=epochs, verbose=1)

    return history


if __name__ == '__main__':
    history = train_dnn(0.2, 20, 200)

    print(history)