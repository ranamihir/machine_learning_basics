import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

X_train = np.vstack((
    0.75*np.random.randn(150, 2) + np.array([1.,0.]),
    0.25*np.random.randn(50, 2) + np.array([-0.5,0.5]),
    0.5*np.random.randn(50, 2) + np.array([-0.5,-0.5]),
))

X_test = np.vstack((
    0.5*np.random.randn(150, 2) + np.array([1.,0.]),
    0.1*np.random.randn(50, 2) + np.array([0.,1.]),
    1*np.random.randn(50, 2) + np.array([-1.,-1.]),
))

class KMeans():
    def __init__(self, n_clusters=3, init='random', n_init=5, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self._check_init()
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        centroids = []
        for init in range(self.n_init):
            self.centroids = self._init_centroids(X)
            i = 1

            while i <= self.max_iter:
                closest = self._get_closest_centroid(X)
                new_centroids = self._update_centroids(X, closest)

                if np.sqrt(np.mean((new_centroids-self.centroids)**2)) < self.tol:
                    break

                self.centroids = new_centroids
                i += 1
            centroids.append(self.centroids)
        self.centroids = mode(np.array(centroids), axis=0)[0][0]
        return self

    def predict(self, X):
        return self._get_closest_centroid(X)

    def _get_closest_centroid(self, X):
        '''
        :param X -> (n, d)
        :param self.centroids -> (k, d)
        :param distances -> (n, k, d) -> (n, k)
        :return labels -> (n,)
        '''

        # Option 1:
        # distances = np.array([np.sqrt((X - self.centroids[c])**2) for c in range(self.n_clusters)]).sum(axis=2)
        # return np.argmin(distances, axis=0)

        # Option 2:
        # distances = np.sqrt((X - self.centroids[:,np.newaxis,:])**2).sum(axis=2)
        # return np.argmin(distances, axis=0)

        # Option 3:
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :])**2, axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, closest):
        return np.array([np.mean(X[closest == c], axis=0) for c in range(self.n_clusters)])

    def _init_centroids(self, X):
        if self.init == 'random' or 'kmeans++':
            indices = np.random.choice(range(len(X)), size=self.n_clusters)
            return X[indices]

    def _check_init(self):
        valid_inits = ['random', 'kmeans++']
        # TODO: kmeans++
        if not self.init in valid_inits:
            raise ValueError('Param "init" must be in "{}"'.format(valid_inits))

kmeans = KMeans(n_clusters=3)

kmeans.fit(X_train)

fig = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
ax.cla()
ax.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.predict(X_train), marker='^', alpha=0.5)
ax.scatter(X_test[:, 0], X_test[:, 1], c= kmeans.predict(X_test), marker='o', alpha=0.5)
ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', s=100)
plt.show()