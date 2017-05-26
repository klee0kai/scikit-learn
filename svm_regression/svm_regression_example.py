from sklearn import svm
from numpy import *

X = [[0, 0, 0, 1], [1, 2, 3, 1]]
y = [[0.5, 1], [2.5, 2]]
clf = svm.SVR()
clf.fit(X, y)

print clf.predict([[1, 1, 1]])
