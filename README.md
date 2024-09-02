## Overview
This repository contains a simple decision tree classifier implemented from scratch in Python using NumPy. The code includes a basic implementation of a decision tree for supervised learning, with methods to train and predict based on input data.

## Project Structure
- **`classifier.py`**: The main script containing the decision tree classifier (`DecisionTreeClassifier` class) and a wrapper for it (`Classifier` class). It also includes a `Node` class to represent the individual nodes of the decision tree.

## Features
- **Classifier**: A wrapper class that manages the training and prediction of the decision tree classifier. It includes methods to reset the classifier and fit it to the provided data.
- **Decision Tree Classifier**: A custom implementation of the decision tree algorithm, featuring:
  - Recursive tree growth
  - Gini impurity calculation for splitting
  - A configurable maximum depth to prevent overfitting
- **Node Class**: Represents the nodes within the decision tree, storing information about feature splits and leaf values.

## Class Descriptions

### `Classifier`
A wrapper class for the decision tree classifier, encapsulating its operations.

- `__init__()`: Initializes the decision tree with a maximum depth of 5.
- `reset()`: Resets the classifier to its default parameters.
- `fit(data, target)`: Trains the classifier using the provided data and target labels.
- `predict(data, legal=None)`: Predicts the class for new data inputs.

### `DecisionTreeClassifier`
The core decision tree implementation, responsible for fitting the model to data and making predictions.

- `__init__(max_depth=None)`: Initializes the tree with a specified maximum depth.
- `fit(X, y)`: Fits the tree to the training data `X` and labels `y`.
- `_grow_tree(X, y, depth=3)`: Recursively grows the tree by splitting nodes based on the best feature and threshold.
- `predict(X)`: Predicts labels for the provided input data.
- Internal methods for:
  - Finding the best split (`_find_best_split`)
  - Calculating Gini impurity (`_calculate_gini`, `_gini_impurity`)
  - Traversing the tree for predictions (`_traverse_tree`)
