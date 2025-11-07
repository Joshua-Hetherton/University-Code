"""
Machine Learning Library

This module implements various machine learning algorithms and utilities for educational purposes.
It includes data handling, decision trees, linear classifiers, and evaluation methods.
Designed to help students understand core ML concepts through clear implementations.
"""

import copy
from collections import defaultdict
from statistics import stdev, mean
import random
import heapq

import numpy as np

from utils import (
    mean_boolean_error,
    open_data,
    num_or_str,
    remove_all,
    unique,
    mode,
    normalize,
    sigmoid,
    argmax_random_tie,
)


class DataSet:
    """
    A data set for a machine learning problem. It has the following fields:

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs.
    """

    def __init__(
        self,
        examples=None,
        attrs=None,
        attr_names=None,
        target=-1,
        inputs=None,
        values=None,
        distance=mean_boolean_error,
        name="",
        source="",
        exclude=(),
    ):
        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(open_data(name + ".csv").read())
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs
        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """
        Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet.
        """
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        else:
            self.inputs = [
                a for a in self.attrs if a != self.target and a not in exclude
            ]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))
        if self.got_values_flag:
            # only check if values are provided while initializing DataSet
            list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError(
                        "Bad value {} for attribute {} in {}".format(
                            example[a], self.attr_names[a], example
                        )
                    )

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        """
        Update the possible values for each attribute by examining all examples.
        This method scans through all examples to determine what values each attribute can take.
        Used internally when the dataset is modified or when values weren't provided initially.
        """
        self.values = list(map(unique, zip(*self.examples)))

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [
            attr_i if i in self.inputs else None for i, attr_i in enumerate(example)
        ]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=""):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])

        for v in self.examples:
            # Extract only input features (exclude target attribute)
            item = [v[i] for i in self.inputs]
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def find_means_and_deviations(self):
        """
        Calculate statistical measures for each class in the dataset.
        This is used for Naive Bayes classification with continuous features.

        For each class (target value), this method computes:
        - means: average value of each feature for examples in that class
        - deviations: standard deviation of each feature for examples in that class

        Returns:
            tuple: (means_dict, deviations_dict) where each dict maps class -> list of values

        Example:
            If we have 2 classes and 3 features, means might look like:
            {'spam': [2.1, 5.3, 1.8], 'ham': [1.2, 3.1, 2.5]}
        """
        target_names = self.values[self.target]
        feature_numbers = len(self.inputs)

        item_buckets = self.split_values_by_classes()

        means = defaultdict(lambda: [0] * feature_numbers)
        deviations = defaultdict(lambda: [0] * feature_numbers)

        for t in target_names:
            # find all the item feature values for item in class t
            features = [[] for _ in range(feature_numbers)]
            for item in item_buckets[t]:
                for i in range(feature_numbers):
                    features[i].append(item[i])

            # calculate means and deviations for the class
            for i in range(feature_numbers):
                means[t][i] = mean(features[i])
                # Handle case where there's only one sample (stdev would fail)
                if len(features[i]) > 1:
                    deviations[t][i] = stdev(features[i])
                else:
                    deviations[t][i] = 1e-9  # Small epsilon to avoid division by zero

        return means, deviations

    def __repr__(self):
        return "<DataSet({}): {:d} examples, {:d} attributes>".format(
            self.name, len(self.examples), len(self.attrs)
        )


def parse_csv(input, delim=","):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


def err_ratio(predict, dataset, examples=None):
    """
    Calculate the error rate of a predictor on a dataset.

    This function measures how often the predictor gets the wrong answer.
    It's a key metric for evaluating machine learning model performance.

    Args:
        predict: A function that takes an example and returns a prediction
        dataset: The DataSet object containing the data structure info
        examples: List of examples to test on (defaults to dataset.examples)

    Returns:
        float: Error rate between 0.0 (perfect) and 1.0 (always wrong)

    Example:
        If predictor gets 8 out of 10 examples correct, error rate = 0.2
    """
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = predict(dataset.sanitize(example))
        if output == desired:
            right += 1
    return 1 - (right / len(examples))


def train_test_split(dataset, start=None, end=None, test_split=None):
    """
    Split a dataset into training and testing portions.
    This is essential for proper machine learning evaluation - we train on one set
    and test on another to get an honest assessment of performance.

    Two ways to use this function:
    1. Specify start/end indices: test set = examples[start:end], rest = training
    2. Specify test_split ratio: test_split=0.2 means 20% for testing, 80% for training

    Args:
        dataset: DataSet object to split
        start: Starting index for test set (when using indices method)
        end: Ending index for test set (when using indices method)
        test_split: Fraction of data to use for testing (when using ratio method)

    Returns:
        tuple: (training_examples, test_examples)

    Example:
        train, test = train_test_split(my_dataset, test_split=0.3)  # 70/30 split
    """
    examples = dataset.examples
    if test_split is None:
        train = examples[:start] + examples[end:]
        val = examples[start:end]
    else:
        total_size = len(examples)
        val_size = int(total_size * test_split)
        train_size = total_size - val_size
        train = examples[:train_size]
        val = examples[train_size:total_size]

    return train, val


def cross_validation_wrapper(learner, dataset, k=10, trials=1):
    """
    Find the optimal model size using cross-validation.

    This function automatically searches for the best model complexity by trying
    different sizes and measuring validation error. It stops when performance
    stabilizes, indicating we've found a good size.

    Args:
        learner: A learning algorithm function that takes (dataset, size)
        dataset: The DataSet to use for validation
        k: Number of folds for cross-validation (default: 10)
        trials: Number of times to repeat the process (default: 1)

    Returns:
        A trained model using the optimal size found

    Note: This implements a simple form of hyperparameter tuning.
    """
    errs = []
    size = 1
    while True:
        errT, errV = cross_validation(learner, dataset, size, k, trials)
        errs.append(errV)

        # check for convergence - if we have enough data points and error is stabilizing
        if len(errs) >= 5:
            recent_errs = errs[-5:]
            if max(recent_errs) - min(recent_errs) < 1e-6:  # convergence criterion
                break

        size += 1
        if size > len(dataset.examples):  # safety break
            break

    # Find best size with minimum validation error
    best_size = np.argmin(errs) + 1  # +1 because size starts at 1
    return learner(dataset, best_size)


def cross_validation(learner, dataset, size=None, k=10, trials=1):
    """
    Perform k-fold cross-validation to estimate model performance.

    Cross-validation is a technique to get a more reliable estimate of how well
    a model will perform on unseen data. We split data into k parts, train on k-1,
    test on 1, and repeat k times.

    Args:
        learner: Learning algorithm function
        dataset: DataSet object to validate on
        size: Model size parameter (if applicable)
        k: Number of folds (default: 10, meaning 90% train, 10% validate each round)
        trials: Number of complete cross-validation runs to average over

    Returns:
        tuple: (average_training_error, average_validation_error)

    Why this matters: Single train/test splits can be misleading due to lucky/unlucky
    splits. Cross-validation gives us a more robust performance estimate.
    """
    k = k or len(dataset.examples)
    if trials > 1:
        trial_errT = 0
        trial_errV = 0
        for t in range(trials):
            errT, errV = cross_validation(learner, dataset, size, k, 1)
            trial_errT += errT
            trial_errV += errV
        return trial_errT / trials, trial_errV / trials
    else:
        fold_errT = 0
        fold_errV = 0
        n = len(dataset.examples)
        examples = dataset.examples.copy()  # Make a copy to avoid modifying original
        random.shuffle(examples)

        # Create a temporary dataset with shuffled examples
        temp_dataset = copy.copy(dataset)
        temp_dataset.examples = examples

        for fold in range(k):
            train_data, val_data = train_test_split(
                temp_dataset, fold * (n // k), (fold + 1) * (n // k)
            )
            # Create a new dataset copy for training to avoid modifying temp_dataset
            train_dataset = copy.copy(temp_dataset)
            train_dataset.examples = train_data
            if size is not None:
                h = learner(train_dataset, size)
            else:
                h = learner(train_dataset)
            fold_errT += err_ratio(h, train_dataset, train_data)
            fold_errV += err_ratio(h, train_dataset, val_data)

        return fold_errT / k, fold_errV / k


def leave_one_out(learner, dataset, size=None):
    """
    Perform leave-one-out cross-validation.

    This is an extreme form of cross-validation where k = number of examples.
    For each example, we train on all other examples and test on just that one.
    Very thorough but computationally expensive for large datasets.

    Often used when you have a small dataset and want the most reliable estimate possible.
    """
    return cross_validation(learner, dataset, size, len(dataset.examples))


class DecisionFork:
    """
    A fork (internal node) of a decision tree.

    In decision trees, internal nodes test an attribute and branch based on its value.
    For example, a node might test "Age" and have branches for "Young", "Middle", "Old".
    Each branch leads to either another fork or a leaf (final decision).

    This represents the "if-then" logic: "IF Age=Young THEN go to left subtree"

    Attributes:
        attr: Index of the attribute to test
        attr_name: Human-readable name of the attribute
        default_child: What to return if we see an unknown attribute value
        branches: Dictionary mapping attribute values to subtrees
    """

    def __init__(self, attr, attr_name=None, default_child=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attr_val = example[self.attr]
        if attr_val in self.branches:
            return self.branches[attr_val](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        """
        Print a human-readable representation of this decision tree node.

        Args:
            indent: How many levels deep we are (for pretty printing)
        """
        name = self.attr_name
        print("Test", name)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, name, "=", val, "==>", end=" ")
            subtree.display(indent + 1)

    def __repr__(self):
        return "DecisionFork({0!r}, {1!r}, {2!r})".format(
            self.attr, self.attr_name, self.branches
        )


class DecisionLeaf:
    """
    A leaf (terminal node) of a decision tree.

    Leaves represent final decisions/predictions. When we reach a leaf,
    we stop asking questions and return the result stored here.

    For example, a leaf might contain "Spam" or "Not Spam" for email classification.
    """

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self, indent=0):
        """Print the final result/prediction stored in this leaf."""
        print("RESULT =", self.result)

    def __repr__(self):
        return repr(self.result)


def DecisionTreeLearner(dataset):
    """
    Build a decision tree using the ID3 algorithm.

    Decision trees work by repeatedly asking questions about the data to split it
    into pure groups. This implements the classic ID3 algorithm which uses
    information gain to choose the best questions to ask.

    The algorithm:
    1. If all examples have same class -> make a leaf with that class
    2. If no attributes left -> make a leaf with majority class
    3. Otherwise -> pick best attribute using information gain, split data, recurse

    Args:
        dataset: DataSet object containing training examples

    Returns:
        A decision tree (either DecisionFork or DecisionLeaf) that can classify examples

    This is a classic "greedy" algorithm - it makes locally optimal choices at each step.
    """
    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        """
        Recursive function that builds the decision tree.

        This is the heart of the ID3 algorithm. At each step, we:
        1. Check if we can make a decision (all same class or no examples)
        2. If not, pick the best attribute to split on
        3. Split examples by that attribute's values
        4. Recursively build subtrees for each split
        """
        if len(examples) == 0:
            return plurality_value(parent_examples)
        if all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        if len(attrs) == 0:
            return plurality_value(examples)
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attr_names[A], plurality_value(examples))
        for v_k, exs in split_by(A, examples):
            subtree = decision_tree_learning(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    def plurality_value(examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(
            values[target], key=lambda v: count(target, v, examples)
        )
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

        def entropy(examples):
            if not examples:  # Handle empty examples
                return 0
            return information_content(
                [count(target, v, examples) for v in values[target]]
            )

        n = len(examples)
        if n == 0:  # Handle empty examples
            return 0

        remainder = sum(
            (len(examples_i) / n) * entropy(examples_i)
            for (v, examples_i) in split_by(attr, examples)
        )
        return entropy(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """
    Calculate information content (entropy) of a probability distribution.

    Entropy measures "how much information" or "how much uncertainty" is in a set of values.
    - Low entropy = predictable, organized (e.g., all examples are same class)
    - High entropy = unpredictable, chaotic (e.g., equal mix of all classes)

    Formula: -sum(p_i * log2(p_i)) where p_i is probability of class i

    Args:
        values: List of counts for each class (e.g., [10, 5] means 10 of class 0, 5 of class 1)

    Returns:
        float: Entropy in bits (0 = perfectly predictable, higher = more chaotic)

    Example: [8, 2] has lower entropy than [5, 5] because first is more predictable
    """
    probabilities = normalize(remove_all(0, values))
    if not probabilities:  # Handle empty list
        return 0
    # Add small epsilon to prevent log(0) and ensure numerical stability
    epsilon = 1e-10
    return sum(
        -p * np.log2(p + epsilon) for p in probabilities if p > 0
    )  # Handle zero probabilities


def NearestNeighborLearner(dataset, k=1):
    """
    Create a k-Nearest Neighbor classifier.

    KNN is a "lazy learning" algorithm - it doesn't build a model during training.
    Instead, for each new example to classify:
    1. Find the k most similar examples in the training data
    2. Let those k neighbors "vote" on the classification
    3. Return the majority vote

    Args:
        dataset: Training data
        k: Number of neighbors to consider (default=1)

    Returns:
        A predict function that classifies new examples

    Why it works: Similar examples often have similar classifications.
    The assumption is "birds of a feather flock together."
    """

    def predict(example):
        """Find the k closest items, and have them vote for the best."""
        best = heapq.nsmallest(
            k, ((dataset.distance(e, example), e) for e in dataset.examples)
        )
        return mode(e[dataset.target] for (d, e) in best)

    return predict


def LinearLearner(dataset, learning_rate=0.01, epochs=100):
    """
    Create a linear classifier using gradient descent.
    [Section 18.6.3]

    Linear classifiers find a straight line (or hyperplane in multiple dimensions)
    that separates different classes. The prediction is based on which side of
    the line a new point falls on.

    Mathematical model: prediction = w₀ + w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ
    where w are weights learned from data, x are features

    Args:
        dataset: Training data
        learning_rate: How big steps to take when updating weights (default: 0.01)
        epochs: How many times to go through all training data (default: 100)

    Returns:
        A predict function that outputs raw scores (not probabilities)

    Note: This uses gradient descent to minimize prediction errors.
    """
    idx_i = dataset.inputs
    idx_t = dataset.target
    examples = dataset.examples
    num_examples = len(examples)

    # Build feature matrix X and target vector y
    X = []
    y = []
    for example in examples:
        # Extract only input features and add bias term
        features = [1.0] + [example[i] for i in idx_i]
        X.append(features)
        y.append(example[idx_t])

    # initialize random weights
    num_weights = len(idx_i) + 1
    w = [random.uniform(-0.5, 0.5) for _ in range(num_weights)]

    # Training loop
    for epoch in range(epochs):
        # Compute predictions and errors for all examples
        errors = []
        for i, example_features in enumerate(X):
            # Compute prediction: w · x
            prediction = sum(w[j] * example_features[j] for j in range(len(w)))
            error = y[i] - prediction
            errors.append(error)

        # Update weights using gradient descent
        for j in range(len(w)):
            # Compute gradient for weight j: -sum(error_i * x_i_j) / num_examples
            gradient = 0
            for i in range(num_examples):
                gradient += errors[i] * X[i][j]
            gradient = gradient / num_examples

            # Update weight
            w[j] = w[j] + learning_rate * gradient

    def predict(example):
        # Extract only input features and add bias
        features = [1.0] + [example[i] for i in idx_i]
        # Compute prediction: w · x
        return sum(w[j] * features[j] for j in range(len(w)))

    return predict


def LogisticLinearLearner(dataset, learning_rate=0.01, epochs=100):
    """
    Create a logistic regression classifier.
    [Section 18.6.4]

    Similar to LinearLearner, but uses the sigmoid function to convert raw scores
    into probabilities between 0 and 1. This makes it better for classification
    because outputs are interpretable as "confidence" in the prediction.

    Mathematical model: probability = sigmoid(w₀ + w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ)
    where sigmoid(z) = 1/(1 + e^(-z))

    Args:
        dataset: Training data
        learning_rate: Step size for gradient descent (default: 0.01)
        epochs: Number of training iterations (default: 100)

    Returns:
        A predict function that outputs probabilities between 0 and 1

    Advantage over LinearLearner: Outputs are probabilities, not raw scores.
    """
    idx_i = dataset.inputs
    idx_t = dataset.target
    examples = dataset.examples
    num_examples = len(examples)

    # Build feature matrix X and target vector y
    X = []
    y = []
    for example in examples:
        # Extract only input features and add bias term
        features = [1.0] + [example[i] for i in idx_i]
        X.append(features)
        y.append(example[idx_t])

    # initialize random weights
    num_weights = len(idx_i) + 1
    w = [random.uniform(-0.5, 0.5) for _ in range(num_weights)]

    for epoch in range(epochs):
        # Compute gradients for all examples (batch gradient descent)
        gradients = [0.0] * len(w)

        for i, example_features in enumerate(X):
            # Compute prediction: sigmoid(w · x)
            prediction = sigmoid(sum(w[j] * example_features[j] for j in range(len(w))))
            error = y[i] - prediction

            # Accumulate gradients
            for j in range(len(w)):
                gradients[j] += error * example_features[j]

        # Update weights using accumulated gradients
        for j in range(len(w)):
            w[j] = w[j] + learning_rate * gradients[j] / num_examples

    def predict(example):
        # Extract only input features and add bias
        features = [1.0] + [example[i] for i in idx_i]
        # Compute prediction: sigmoid(w · x)
        return sigmoid(sum(w[j] * features[j] for j in range(len(w))))

    return predict


def init_examples(examples, idx_i, idx_t, o_units):
    """
    Prepare examples for neural network training.

    Neural networks need data in a specific format:
    - Input features separated from target values
    - Targets in "one-hot" format for multi-class problems

    Args:
        examples: Raw training examples
        idx_i: Indices of input features
        idx_t: Index of target attribute
        o_units: Number of output units (classes)

    Returns:
        tuple: (inputs_dict, targets_dict) formatted for neural network training

    One-hot encoding example: If classes are ['cat', 'dog', 'bird'] and target is 'dog',
    then one-hot representation is [0, 1, 0]
    """
    inputs, targets = {}, {}

    for i, e in enumerate(examples):
        # input values of e
        inputs[i] = [e[i] for i in idx_i]

        if o_units > 1:
            # one-hot representation of e's target
            t = [0 for i in range(o_units)]
            t[e[idx_t]] = 1
            targets[i] = t
        else:
            # target value of e
            targets[i] = [e[idx_t]]

    return inputs, targets


# Performance Metrics
# These functions help evaluate how well our machine learning models are doing


def accuracy_score(y_pred, y_true):
    """
    Calculate classification accuracy.

    Accuracy = (Number of correct predictions) / (Total predictions)
    This is the most intuitive metric - what percentage did we get right?

    Args:
        y_pred: Predicted labels (numpy array)
        y_true: True/actual labels (numpy array)

    Returns:
        float: Accuracy between 0.0 (all wrong) and 1.0 (all correct)

    Example: If we predict [1,0,1,1] and truth is [1,0,0,1], accuracy = 3/4 = 0.75
    """
    assert y_pred.shape == y_true.shape
    return np.mean(np.equal(y_pred, y_true))


def r2_score(y_pred, y_true):
    """
    Calculate R² (coefficient of determination) for regression.

    R² measures how well our predictions explain the variance in the data.
    - R² = 1.0 means perfect predictions
    - R² = 0.0 means we're no better than just predicting the average
    - R² < 0.0 means we're worse than predicting the average

    Formula: R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals, SS_tot = total sum of squares

    Args:
        y_pred: Predicted values (numpy array)
        y_true: True/actual values (numpy array)

    Returns:
        float: R² score (higher is better, 1.0 is perfect)

    Used for: Evaluating regression models (predicting continuous values)
    """
    assert y_pred.shape == y_true.shape
    return 1.0 - (
        np.sum(np.square(y_pred - y_true))  # sum of square of residuals
        / np.sum(np.square(y_true - np.mean(y_true)))
    )  # total sum of squares


# Sample Datasets for Learning and Testing
# These provide ready-to-use datasets for experimenting with machine learning algorithms

# O-ring dataset: Predicts Space Shuttle O-ring failures based on temperature and pressure
# Historical significance: Related to Challenger disaster analysis
orings = DataSet(
    name="orings",
    target="Distressed",
    attr_names="Rings Distressed Temp Pressure Flightnum",
)

# Zoo dataset: Classify animals into categories based on their characteristics
# Good for decision tree learning - features like "has hair", "lays eggs", etc.
zoo = DataSet(
    name="zoo",
    target="type",
    exclude=["name"],  # Don't use animal name as a feature (that would be cheating!)
    attr_names="name hair feathers eggs milk airborne aquatic predator toothed backbone "
    "breathes venomous fins legs tail domestic catsize type",
)

# Iris dataset: Classic ML dataset for flower classification
# Famous for being linearly separable and having clean, numerical features
iris = DataSet(
    name="iris",
    target="class",
    attr_names="sepal-len sepal-width petal-len petal-width class",
)


def RestaurantDataSet(examples=None):
    """
    Build a DataSet for the classic "Restaurant Waiting" problem.

    This is a standard example in AI textbooks for learning decision trees.
    The goal is to decide whether to wait for a table at a restaurant
    based on various factors like how busy it is, if you have a reservation, etc.

    Features include: alternate restaurant available, bar area, day of week,
    how hungry you are, number of patrons, price range, weather, etc.
    """
    return DataSet(
        name="restaurant",
        target="Wait",
        examples=examples,
        attr_names="Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Type WaitEstimate Wait",
    )


restaurant = RestaurantDataSet()


def T(attr_name, branches):
    """
    Helper function to build decision tree nodes more easily.

    This makes it simpler to construct decision trees by hand for testing.
    It automatically converts string values to DecisionLeaf objects.

    Args:
        attr_name: Name of attribute to test at this node
        branches: Dictionary mapping attribute values to subtrees or final decisions

    Returns:
        DecisionFork object representing this node
    """
    branches = {
        value: (child if isinstance(child, DecisionFork) else DecisionLeaf(child))
        for value, child in branches.items()
    }
    # Use a proper default child instead of print function
    default_child = DecisionLeaf(
        "No"
    )  # Default decision when attribute value is unknown
    return DecisionFork(
        restaurant.attr_num(attr_name), attr_name, default_child, branches
    )


""" 
Example Decision Tree: Restaurant Waiting Decision

This tree encodes human decision-making logic for whether to wait at a restaurant.
It demonstrates how decision trees can capture complex, nested decision processes
that humans use naturally.

The tree asks questions like:
- How many patrons are there? (None/Some/Full)  
- How long is the wait? (0-10 min, 10-30 min, etc.)
- Is there an alternate restaurant? (Yes/No)

This serves as a "ground truth" for testing decision tree learning algorithms.
"""

waiting_decision_tree = T(
    "Patrons",
    {
        "None": "No",
        "Some": "Yes",
        "Full": T(
            "WaitEstimate",
            {
                ">60": "No",
                "0-10": "Yes",
                "30-60": T(
                    "Alternate",
                    {
                        "No": T(
                            "Reservation",
                            {"Yes": "Yes", "No": T("Bar", {"No": "No", "Yes": "Yes"})},
                        ),
                        "Yes": T("Fri/Sat", {"No": "No", "Yes": "Yes"}),
                    },
                ),
                "10-30": T(
                    "Hungry",
                    {
                        "No": "Yes",
                        "Yes": T(
                            "Alternate",
                            {
                                "No": "Yes",
                                "Yes": T("Raining", {"No": "No", "Yes": "Yes"}),
                            },
                        ),
                    },
                ),
            },
        ),
    },
)


def SyntheticRestaurant(n=20):
    """
    Generate artificial restaurant data using the decision tree above.

    This creates synthetic training data by:
    1. Generating random combinations of restaurant features
    2. Using our hand-built decision tree to determine the "correct" decision
    3. Returning a dataset that learning algorithms can train on

    Args:
        n: Number of examples to generate (default: 20)

    Returns:
        DataSet with n artificially generated restaurant examples

    Useful for: Testing whether learning algorithms can recover the original
    decision tree from the data it generates.
    """

    def gen():
        example = list(map(random.choice, restaurant.values))
        example[restaurant.target] = waiting_decision_tree(example)
        return example

    return RestaurantDataSet([gen() for _ in range(n)])
