# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np


class Classifier:
    """
    Classifier wrapper that encapsulates the decision tree classifier.
    """

    def __init__(self):
        self.clf = DecisionTreeClassifier(max_depth=5)

    def reset(self):
        """Reset the classifier to default parameters."""
        self.clf = DecisionTreeClassifier(max_depth=5)

    def fit(self, data, target):
        """
        Fit the classifier with data and target labels.

        Parameters:
        data : array-like, Training data.
        target : array-like, Target labels for training data.
        """
        data = np.array(data)
        target = np.array(target)
        self.clf.fit(data, target)

    def predict(self, data, legal=None):
        """
        Predict the action for the given data.

        Parameters:
        data : array-like, Data for prediction.
        legal : array-like, optional, Legal actions. Default is None.
        """
        features = np.array(data).reshape(1, -1)
        action = self.clf.predict(features)
        return action


class Node:
    """
    Class representing a node in the decision tree.

    Parameters:
    feature_index : int, optional, Index of the feature this node splits on. Default is None.
    threshold : float, optional, Threshold value for the feature. Default is None.
    left : Node, optional,  Left child node. Default is None.
    right : Node, optional,  Right child node. Default is None.
    value : int, optional,  Predicted class value for leaf nodes. Default is None.
    """

    def __init__(self, feature_index=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    """
    Decision tree classifier.

    Parameters:
    max_depth : int, optional,  Maximum depth of the decision tree. Default is None.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Train the decision tree classifier.

        Parameters:
        X : array-like of shape (n_samples, n_features), Training samples.
        y : array-like of shape (n_samples,), True labels for samples.
        """
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=3):
        """
        Recursively grow the decision tree.

        Parameters:
        X : array-like of shape (n_samples, n_features), Training samples for the subtree.
        y : array-like of shape (n_samples,), Labels for the subtree samples.
        depth : int, optional, Current depth of the node. Default is 3.
        """
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Return a leaf node if stop conditions are met
        if (
                self.max_depth is not None and depth >= self.max_depth) or num_classes == 1 or num_samples < 2:
            return Node(value=np.bincount(y).argmax())

        # Find the best split
        best_feature_index, best_threshold = self._find_best_split(X, y)

        # Create nodes and subtrees based on the best split
        left_indices = X[:, best_feature_index] < best_threshold
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[~left_indices], y[~left_indices], depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold,
                    left=left, right=right)

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters:
        X : array-like of shape (n_samples, n_features), Training samples.
        y : array-like of shape (n_samples,), Sample labels.
        """
        num_samples, num_features = X.shape
        best_gini = float('inf')
        best_feature_index, best_threshold = None, None
        # Iterate through all features
        for feature_index in range(num_features):
            # Get possible thresholds for this feature
            thresholds = np.unique(X[:, feature_index])
            # Try all thresholds
            for threshold in thresholds:
                # Split samples based on threshold
                left_indices = X[:, feature_index] < threshold
                # Calculate gini impurity
                gini = self._calculate_gini(y, left_indices)

                # Update best split if better gini found
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _calculate_gini(self, y, left_indices):
        """
        Calculate the weighted Gini impurity.

        Parameters: 
        y : array-like of shape (n_samples,), Sample labels
        left_indices : array-like of shape (n_samples,), Boolean index representing left child samples.
        """
        num_samples = len(y)
        left_gini = self._gini_impurity(y[left_indices])
        right_gini = self._gini_impurity(y[~left_indices])
        total_gini = (np.sum(left_indices) / num_samples) * left_gini + (
                np.sum(~left_indices) / num_samples) * right_gini
        return total_gini

    def _gini_impurity(self, y):
        """
        Calculate the Gini impurity.

        Parameters: 
        y : array-like of shape (n_samples,), Sample labels
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict(self, X):
        """
        Predict the class labels for samples.

        Parameters: 
        X : array-like of shape (n_samples, n_features), Samples to predict.
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """
        Recursively traverse the decision tree for prediction.

        Parameters: 
        x : array-like of shape (n_features,), Features of a single sample.
        node : Node, Current node.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
