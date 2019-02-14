import numpy as np
from collections import Counter
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt


class DecisionTree():
    def __init__(self, criterion, leaf_value_estimator, depth=0,
                 min_samples_split=5, min_samples_leaf=1, \
                 max_features=None, max_depth=3):
        '''
        Base Decision Tree class

        :param criterion: method for splitting node
        :param leaf_value_estimator: method for estimating leaf value
        :param depth: depth indicator, default value is 0, representing root node
        :param min_samples_split: an internal node can be splitted only if it contains \
                                  points more than min_samples_split
        :param min_samples_leaf: minimum number of samples required to be at a leaf node
        :param max_depth: restriction of tree depth.
        '''
        self.criterion = criterion
        self.leaf_value_estimator = leaf_value_estimator
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth

        valid_max_features = ['auto', 'sqrt', None]
        assert isinstance(max_features, int) or max_features in valid_max_features, \
               'Param "max_features" must either be an integer or one of "{}"'.format(valid_max_features)

    def get_params(self, deep=True):
        return vars(self)

    def fit(self, X, y):
        '''
        Fits the tree by setting the values self.is_leaf,
        self.split_id (the index of the feature we want to split on, if we're splitting),
        self.split_value (the corresponding value of that feature where the split is),
        and self.value, which is the prediction value if the tree is a leaf node.

        :param X: a numpy array of training data, shape = (n, m)
        :param y: a numpy array of labels, shape = (n, 1)
        :return self
        '''
        num_instances, num_features = X.shape
        best_index, best_split_value, best_impurity, best_partitions = np.inf, np.inf, np.inf, None

        # Splitting criterion function
        criterion_dict = {
            'entropy': self._compute_entropy,
            'gini': self._compute_gini,
            'mse': np.var,
            'mae': self._mean_absolute_deviation_around_median
        }

        # Node value prediction function
        leaf_value_estimator_dict = {
            'mean': np.mean,
            'median': np.median,
            'mode': self._most_common_label,
        }

        criterion = criterion_dict[self.criterion]
        leaf_value_estimator = leaf_value_estimator_dict[self.leaf_value_estimator]

        num_features = X.shape[1]
        max_features = self._get_max_features(num_features)
        feature_indices = np.random.choice(range(num_features), size=max_features)

        for feat_index in feature_indices:
            for row in X:
                partitions = self._test_split(feat_index, row[feat_index], X)
                impurity = self._compute_impurity(partitions, y, criterion)
                if impurity < best_impurity:
                    best_index, best_split_value, best_impurity, best_partitions = \
                    feat_index, row[feat_index], impurity, partitions

        self.is_leaf = False
        left_partition, right_partition = best_partitions
        total_size = left_partition.shape[0] + right_partition.shape[0]

        # Check for stopping conditions
        if (min(left_partition.shape[0], right_partition.shape[0]) < self.min_samples_leaf) or \
            (total_size < self.min_samples_split)  or (self.depth >= self.max_depth):
            self.is_leaf = True

        if self.is_leaf:
            self.value = leaf_value_estimator(y)
            return self

        # Split at Node
        self.split_id = best_index
        self.split_value = best_split_value

        # Process left child
        self.left = DecisionTree(self.criterion, self.leaf_value_estimator, \
                                  self.depth+1, self.min_samples_split, self.min_samples_leaf, self.max_depth)
        self.left.fit(X[left_partition.astype(int)], y[left_partition.astype(int)])


        # Process right child
        self.right = DecisionTree(self.criterion, self.leaf_value_estimator, \
                                   self.depth+1, self.min_samples_split, self.min_samples_leaf, self.max_depth)
        self.right.fit(X[right_partition.astype(int)], y[right_partition.astype(int)])

        return self

    def predict(self, X):
        '''
        Predict targets by decision tree

        :param X: a numpy array with new data, shape (n, m)
        :return whatever is returned by leaf_value_estimator for leaf containing X
        '''
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        if self.is_leaf:
            return self.value
        if x[self.split_id] <= self.split_value:
            return self.left._predict_instance(x)
        return self.right._predict_instance(x)

    def _test_split(self, index, value, X):
        left = np.where(X[:,index] <= value)[0]
        right = np.where(X[:,index] > value)[0]
        return left, right

    def _compute_entropy(self, label_array):
        '''
        Calulate the entropy of given label list

        :param label_array: a numpy array of labels shape = (n, 1)
        :return entropy: entropy value
        '''
        num_labels = label_array.shape[0]
        entropy = 0.
        for label in np.unique(label_array):
            num_label = label_array[label_array==label].shape[0]
            p_label = num_label/num_labels
            entropy -= p_label*np.log2(p_label)
        return entropy

    def _compute_gini(self, label_array):
        '''
        Calulate the gini index of label list

        :param label_array: a numpy array of labels shape = (n, 1)
        :return gini: gini index value
        '''
        num_labels = label_array.shape[0]
        gini = 1.
        for label in np.unique(label_array):
            num_label = label_array[label_array==label].shape[0]
            p_label = num_label/num_labels
            gini -= p_label**2
        return gini

    def _compute_impurity(self, partitions, label_array, impurity_func):
        '''
        Calculate the impurity for a partitioned dataset
        '''

        # Count all samples at split point
        total_size = sum([partition.shape[0] for partition in partitions])
        # Sum weighted impurities for each partition
        impurity = 0.
        for partition in partitions:
            partition_size = partition.shape[0]
            # Avoid division by zero
            if partition_size:
                # Compute weighted sum of parition impurities by their relative size
                impurity += impurity_func(label_array[partition.astype(int)]) * (partition_size / total_size)
        return impurity

    def _mean_absolute_deviation_around_median(self, y):
        '''
        Calulate the mean absolute deviation around the median of a given target list

        :param y: a numpy array of targets shape = (n, 1)
        :return mae
        '''
        median = np.median(y)
        mae = np.sum(np.abs(y - median))
        return mae

    def _most_common_label(self, y):
        '''
        Find most common label
        '''
        label_cnt = Counter(y.reshape(len(y)))
        label = label_cnt.most_common(1)[0][0]
        return label

    def _get_max_features(self, num_features):
        max_features = self.max_features
        if self.max_features == 'auto':
            max_features = np.round(np.sqrt(num_features)).astype(int)
        elif self.max_features == 'log2':
            max_features = np.round(np.log2(num_features)).astype(int)
        elif self.max_features is None:
            max_features = num_features
        return min(max_features, num_features)


class ClassificationTree():
    def __init__(self, criterion='entropy', min_samples_split=5, \
                 min_samples_leaf=1, max_features=None, max_depth=3):
        '''
        Classification Tree class
        :param criterion(str): loss function for splitting internal node
        '''
        self.tree = DecisionTree(criterion=criterion, leaf_value_estimator='mode', \
                                 depth=0, min_samples_split=min_samples_split, \
                                 min_samples_leaf=min_samples_leaf, \
                                 max_features=max_features, max_depth=max_depth)

    def get_params(self, deep=True):
        return self.tree.get_params(deep)

    def fit(self, X, y):
        self.tree.fit(X, y)
        return self.tree

    def predict(self, X):
        return self.tree.predict(X)


class RegressionTree():
    def __init__(self, criterion='mse', estimator='mean', min_samples_split=5, \
                 min_samples_leaf=1, max_features=None, max_depth=5):
        '''
        Regression Tree class
        :param criterion(str): loss function used for splitting internal nodes
        :param estimator(str): value estimator of internal node
        '''

        self.tree = DecisionTree(criterion=criterion, leaf_value_estimator=estimator, \
                                 depth=0, min_samples_split=min_samples_split, \
                                 min_samples_leaf=min_samples_leaf, \
                                 max_features=max_features, max_depth=max_depth)

    def get_params(self, deep=True):
        return self.tree.get_params(deep)

    def fit(self, X, y):
        self.tree.fit(X, y)
        return self.tree

    def predict(self, X):
        return self.tree.predict(X)


def main():
    ############### Classifiers ###############
    data_train = np.loadtxt('data/svm-train.txt')
    data_test = np.loadtxt('data/svm-test.txt')
    x_train, y_train = data_train[:, 0:2], data_train[:, 2].reshape(-1, 1)
    x_test, y_test = data_test[:, 0:2], data_test[:, 2].reshape(-1, 1)
    y_train_label = (y_train > 0).astype(int).reshape(-1, 1)

    # Plotting decision regions
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, depth, tt in zip(product([0, 1], [0, 1, 2]),
                              range(1,7),
                              ['max_depth = {}'.format(n) for n in range(1, 7)]):

        # Decision Tree Classifier
        dtree = ClassificationTree(max_depth=depth)
        dtree.fit(x_train, y_train_label)

        Z = dtree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        f.suptitle('Decision Tree Classifier')
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(x_train[:, 0], x_train[:, 1], c=y_train_label.ravel(), alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()
    #######################################################

    ############### Regressors ###############
    data_krr_train = np.loadtxt('data/krr-train.txt')
    data_krr_test = np.loadtxt('data/krr-test.txt')
    x_krr_train, y_krr_train = data_krr_train[:,0].reshape(-1,1),data_krr_train[:,1].reshape(-1,1)
    x_krr_test, y_krr_test = data_krr_test[:,0].reshape(-1,1),data_krr_test[:,1].reshape(-1,1)

    plot_size = 0.001
    x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

    f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

    for idx, depth, tt in zip(product([0, 1], [0, 1, 2]),
                              range(1,7),
                              ['max_depth = {}'.format(n) for n in range(1, 7)]):

        # Decision Tree Regressor
        dtree = RegressionTree(max_depth=depth, min_samples_split=1, \
                               criterion='mae', estimator='median')
        dtree.fit(x_krr_train, y_krr_train)

        y_predict = dtree.predict(x_range).reshape(-1, 1)

        f.suptitle('Decision Tree Regressor')
        axarr[idx[0], idx[1]].plot(x_range, y_predict, color='r')
        axarr[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)
        axarr[idx[0], idx[1]].set_xlim(0, 1)

    plt.show()
    #######################################################

if __name__ == '__main__':
    main()
