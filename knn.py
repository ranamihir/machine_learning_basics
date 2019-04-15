import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
import matplotlib.pyplot as plt

X_train = np.vstack((
    0.75*np.random.randn(150, 2) + np.array([1.,0.]),
    0.25*np.random.randn(50, 2) + np.array([-0.5,0.5]),
    0.5*np.random.randn(50, 2) + np.array([-0.5,-0.5]),
))
y_train = np.array([0]*150+[1]*50+[2]*50)

X_test = np.vstack((
    0.5*np.random.randn(150, 2) + np.array([1.,0.]),
    0.1*np.random.randn(50, 2) + np.array([0.,1.]),
    1*np.random.randn(50, 2) + np.array([-1.,-1.]),
))
y_test = np.array([0]*150+[1]*50+[2]*50)

class kNN():
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        # Option 1:
        # distances = cdist(self.X, X, metric='euclidean')
        # idx = np.argpartition(distances, self.k, axis=0)[:self.k]

        # Option 2:
        distances = np.array([np.sqrt(np.sum((self.X - x)**2, axis=1)) for x in X])
        idx = np.argpartition(distances, self.k, axis=1)[:self.k]

#         nearest_distances = np.take(self.y, idx)
        nearest_distances = self.y[idx]
        return mode(nearest_distances, axis=0)[0]

k = 3
knn = kNN(k)

knn.fit(X_train, y_train)

knn.predict(X_train)

knn.predict(X_test)
