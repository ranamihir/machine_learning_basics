import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_features(train, test):
    maxima = np.max(train, 0)
    minima = np.min(train, 0)

    # Remove constant columns
    constant_cols = np.where(maxima == minima)
    train_normalized, test_normalized = np.delete(train, constant_cols, 1), np.delete(test, constant_cols, 1)
    maxima, minima = np.delete(maxima, constant_cols), np.delete(minima, constant_cols)

    # Scale each feature
    train_normalized = (train_normalized - minima)/(maxima - minima)
    test_normalized = (test_normalized - minima)/(maxima - minima)

    return train_normalized, test_normalized

print('\rLoading the dataset... ', end='')
X_train = pd.read_csv('data/X_train.txt', delimiter=',', header=None)
X_val = pd.read_csv('data/X_val.txt', delimiter=',', header=None)
y_train = np.squeeze(pd.read_csv('data/y_train.txt', delimiter=',', header=None).values)
y_val = np.squeeze(pd.read_csv('data/y_val.txt', delimiter=',', header=None).values)
print('Done.')

print("Scaling all to [0, 1]")
X_train, X_val = normalize_features(X_train, X_val)
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1)))) # Add bias term

class LogisticRegression():
    def __init__(self, C=1.0, lr=1e-4, n_epochs=1000, batch_size=64, eps=1e-4):
        self.lambda_reg = 1/C if C != 0. else 0.
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eps = eps

    def fit(self, X, y):
        self.theta = self._init_theta(X)

        p = np.random.permutation(range(len(X)))
        X_shuffled, y_shuffled = X[p], y[p]

        for epoch in range(1, self.n_epochs+1):
            for step in range(0, len(X), self.batch_size):
                X_batch, y_batch = X_shuffled[step:step+self.batch_size], y_shuffled[step:step+self.batch_size]
                grad = self._compute_gradient(X_batch, y_batch)
                self.theta -= self.lr * grad
            loss = self._compute_loss(X_batch, y_batch)
            print(loss)
            if np.sqrt(np.dot(self.theta, self.theta)) < self.eps:
                print('Stopping Early after {} epochs.'.format(epoch))
                break

        return self

    def predict(self, X, threshold=0.5):
        return self._stable_sigmoid(np.dot(X, self.theta)) >= threshold

    def predict_proba(self, X):
        return self._stable_sigmoid(np.dot(X, self.theta))

    def _compute_loss(self, X, y, logscale=True, reg=False):
        prob1 = self.predict_proba(X)
        loss = - np.mean((y*np.log(prob1) + (1-y)*np.log(1-prob1)))
        if reg:
            loss += self.lambda_reg * np.dot(self.theta, self.theta)
        if logscale:
            loss = np.log1p(loss)
        return loss

    def _compute_gradient(self, X, y):
        prob1 = self.predict_proba(X)
        return 2*np.dot(X.T, (prob1 - y))/X.shape[0]  + 2*self.lambda_reg*self.theta

    def _init_theta(self, X):
        return np.random.randn(X.shape[1])

    def _sigmoid(self, x):
        return 1./(1 + np.exp(-x))

    def _stable_sigmoid(self, x):
        z = np.zeros_like(x)
        x_pos, x_neg = np.where(x >= 0)[0], np.where(x < 0)[0]
        z[x_pos] = 1./(1. + np.exp(-x[x_pos]))
        z_neg = np.exp(x[x_neg])
        z[x_neg] = z_neg/(1 + z_neg)
        return z

lr = LogisticRegression(lr=0.05, batch_size=200, n_epochs=100, C=0.)

lr.fit(X_train, y_train)
