import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 3, 2])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, Y)

print(clf.predict_proba([[-0.8, -1]]))

