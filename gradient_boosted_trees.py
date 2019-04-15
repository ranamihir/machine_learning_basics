import numpy as np
from itertools import product
from copy import deepcopy
from decision_trees import ClassificationTree, RegressionTree
import matplotlib.pyplot as plt


class GradientBoostedTree():
    '''
    Gradient Boosting Regressor/Classifier class
    :method fit: fitting model
    '''
    def __init__(self, n_estimators=10, pseudo_residual_func='l2', lr=0.1, \
                 criterion='gini', estimator='mode', min_samples_split=5, \
                 min_samples_leaf=1, max_features='auto', max_depth=3):
        '''
        Initialize gradient boosting class

        :param n_estimators: number of estimators (i.e. number of rounds of gradient boosting)
        :pseudo_residual_func: function used for computing pseudo-residual
        :param lr: step size of gradient descent
        '''

        # Pseudo-residual function for gradient boosting
        pseudo_residual_func_dict = {
            'l2': self._pseudo_residual_L2,
            'logistic': self._pseudo_residual_logistic
        }

        estimator_criterion_dict = {
            'mean': ['mse', 'mae'],
            'median': ['mse', 'mae'],
            'mode': ['gini', 'entropy']
        }

        assert estimator in estimator_criterion_dict.keys(), \
            'Param "estimator" must be one of {}'.format(estimator_criterion_dict.keys())
        assert criterion in estimator_criterion_dict[estimator], \
            'Param "criterion" must be one of {} for leaf value estimator "{}"'\
            .format(estimator_criterion_dict[estimator], estimator)

        valid_max_features = ['auto', 'sqrt', None]
        assert isinstance(max_features, int) or max_features in valid_max_features, \
               'Param "max_features" must either be an integer or one of "{}"'.format(valid_max_features)

        self.n_estimators = n_estimators
        self.pseudo_residual_func = pseudo_residual_func_dict[pseudo_residual_func]
        self.lr = lr
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth

        if estimator in ['mean', 'median']:
            base_estimator = RegressionTree(criterion=criterion, estimator=estimator, max_depth=max_depth, \
                                            min_samples_split=min_samples_split, max_features=max_features, \
                                            min_samples_leaf=min_samples_leaf)
        else:
            base_estimator = ClassificationTree(criterion=criterion, max_depth=max_depth, \
                                                min_samples_split=min_samples_split, max_features=max_features, \
                                                min_samples_leaf=min_samples_leaf)
        self.estimator = estimator
        self.estimators = [deepcopy(base_estimator) for _ in range(n_estimators)]

    def get_params(self, deep=True):
        return self.estimators[0].get_params(deep)

    def fit(self, X, y):
        '''
        Fit gradient boosting model
        '''
        predictions = np.zeros(y.shape[0])
        for i in range(self.n_estimators):
            pseudo_residuals = self.pseudo_residual_func(y, predictions)
            self.estimators[i].fit(X, pseudo_residuals)
            predictions += self.lr * self.estimators[i].predict(X)
        return self

    def predict(self, X):
        '''
        Predict value
        '''
        predictions = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            predictions += self.lr * self.estimators[i].predict(X)
        return predictions

    def _pseudo_residual_L2(self, y, yhat):
        '''
        Compute the pseudo-residual based with square loss on current predicted value.
        '''
        return y - yhat

    def _pseudo_residual_logistic(self, y, yhat):
        '''
        Compute the pseudo-residual based with logistic loss on current predicted value.
        '''
        return y/(1 + np.exp(np.multiply(y, yhat)))


def main():
    ############### Classifiers ###############
    data_train = np.loadtxt('data/svm-train.txt')
    x_train, y_train = data_train[:, 0:2], data_train[:, 2].reshape(-1, 1)
    y_train_label = (y_train > 0).astype(int).reshape(-1, 1)

    # Plotting decision regions
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, n_est, tt in zip(product([0, 1], [0, 1, 2]),
                              [1, 5, 10, 20, 50, 100],
                              ['n_estimators = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):

        # Gradient Boosted Decision Tree Classifier
        gbt = GradientBoostedTree(n_estimators=n_est, pseudo_residual_func='logistic', \
                                  criterion='entropy', estimator='mode', max_depth=5)
        gbt.fit(x_train, y_train.ravel())

        Z = (gbt.predict(np.c_[xx.ravel(), yy.ravel()]) > 0).astype(int)
        Z = Z.reshape(xx.shape)

        f.suptitle('GBDT Classifier')
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(x_train[:, 0], x_train[:, 1], c=y_train_label.ravel(), alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()
    #######################################################

    ############### Regressors ###############
    data_krr_train = np.loadtxt('data/krr-train.txt')
    x_krr_train, y_krr_train = data_krr_train[:,0].reshape(-1,1), data_krr_train[:,1].reshape(-1,1)

    plot_size = 0.001
    x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, n_est, tt in zip(product([0, 1], [0, 1, 2]),
                              [1, 5, 10, 20, 50, 100],
                              ['n_estimators = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):

        # Gradient Boosted Decision Tree Regressor
        gbt = GradientBoostedTree(n_estimators=n_est, pseudo_residual_func='l2', \
                                  criterion='mse', estimator='mean', max_depth=2)
        gbt.fit(x_krr_train, y_krr_train.ravel())

        y_predict = gbt.predict(x_range)

        f.suptitle('GBDT Regressor')
        axarr[idx[0], idx[1]].plot(x_range, y_predict, color='r')
        axarr[idx[0], idx[1]].scatter(x_krr_train, y_krr_train.ravel(), alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
        axarr[idx[0], idx[1]].set_xlim(0, 1)

    plt.show()
    #######################################################

if __name__ == '__main__':
    main()
