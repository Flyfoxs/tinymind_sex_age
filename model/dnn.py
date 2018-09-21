import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

from tiny.tfidf import *
from tiny.usage import *


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

model.add(Dense(20))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(22))
model.add(Activation('softmax'))

# model.compile(optimizer="sgd", loss="mse")
from keras.metrics import categorical_accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])


model.fit(X_train, np_utils.to_categorical(y_train), epochs=10, verbose=1)

from sklearn.metrics import log_loss

best = log_loss(y_test, model.predict_proba(X_test) )

best