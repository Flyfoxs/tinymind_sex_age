# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
from keras.utils import np_utils

def convert2num(category):
    import numpy as np
    import pandas as pd
    category = np.array(category)
    a_enc = pd.factorize(category)
    print(f'{len(a_enc[1])}:{sorted(a_enc[1])}')
    return a_enc[0]

DATA_DIR = "./cache"

file_name = 'get_device_app_sequence_mini.csv'
#file_name = 'get_device_app_sequence_[]_{}.csv'


MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 1000

# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
ftrain = open(os.path.join(DATA_DIR, file_name), 'r')

print('Begin to count')
for line in ftrain:
    label, sentence = line.strip().split(",")
    if label =='' or label=='sex_age':
        continue
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
    if num_recs % 1000 == 0 :
        print(f'{"="*20}_{num_recs}')

ftrain.close()

print(f'count work, row:{num_recs}')

## Get some information about our corpus
#print maxlen            # 42
#print len(word_freqs)   # 2313

# 1 is UNK, 0 is PAD
# We take MAX_FEATURES-1 featurs to accound for PAD
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in
                enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

# convert sentences to sequences
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
ftrain = open(os.path.join(DATA_DIR, file_name), 'r')
label_list=[]
for line in ftrain:
    label, sentence = line.strip().split(",")
    if label =='' or label=='sex_age':
        continue
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        #print(type(word2index))
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    label_list.append(label)
    #y[i] = int(label)
    i += 1
ftrain.close()
y = convert2num(label_list)

print(y.shape)
y = np_utils.to_categorical(y, 22)


# Pad the sequences (left padded with zeros)
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# Build model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,
                    input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(22))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              #loss="binary_crossentropy", optimizer="adam",
              #metrics=["accuracy"]
              )

# check_best = ModelCheckpoint(filepath=replace_invalid_filename_char(file_path),
#                              monitor='val_loss', verbose=1,
#                              save_best_only=True, mode='min')

early_stop = EarlyStopping(monitor='val_loss', verbose=1,
                           patience=300,
                           )


history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(Xtest, ytest))

# # plot loss and accuracy
# plt.subplot(211)
# plt.title("Accuracy")
# plt.plot(history.history["acc"], color="g", label="Train")
# plt.plot(history.history["val_acc"], color="b", label="Validation")
# plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# evaluate
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("%.0f\t%d\t%s" % (ypred, ylabel, sent))


