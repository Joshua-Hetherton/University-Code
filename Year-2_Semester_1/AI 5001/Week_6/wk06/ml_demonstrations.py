"""
Machine Learning Techniques Demonstration

This file demonstrates all the machine learning techniques available in learning.py. It includes:
1. Decision Tree Learning
2. k-Nearest Neighbors (kNN)
3. Linear Classifiers (Linear and Logistic Regression)
4. Cross-validation techniques
5. Performance evaluation metrics
"""

import numpy as np
import random
from learning import (
    # Datasets
    iris,
    zoo,
    orings,
    restaurant,
    SyntheticRestaurant,
    # Learning algorithms
    DecisionTreeLearner,
    NearestNeighborLearner,
    LinearLearner,
    LogisticLinearLearner,
    # Evaluation functions
    cross_validation,
    leave_one_out,
    train_test_split,
    err_ratio,
    accuracy_score,
    r2_score,
    # Helper classes
    DataSet,
)


def print_separator(title=""):
    """Print a nice separator for organizing output."""
    print("\n" + "=" * 60)
    if title:
        print(f" {title} ")
        print("=" * 60)
    else:
        print()


def demonstrate_decision_trees():
    """Demonstrate decision tree learning on multiple datasets."""
    print_separator("DECISION TREE LEARNING")

    # 1. Zoo Dataset - Perfect for decision trees (categorical features)
    print("1. Zoo Dataset Classification")
    print(f"Dataset info: {zoo}")
    print(f"Target attribute: {zoo.attr_names[zoo.target]}")
    print(
        f"Features: {[zoo.attr_names[i] for i in zoo.inputs[:5]]}... (showing first 5)"
    )

    # Build decision tree
    zoo_tree = DecisionTreeLearner(zoo)

    # Test on training data
    zoo_error = err_ratio(zoo_tree, zoo)
    print(f"Training error rate: {zoo_error:.3f}")
    print(f"Training accuracy: {(1 - zoo_error) * 100:.1f}%")

    # Display part of the tree structure
    print("\nDecision tree structure (first few levels):")
    zoo_tree.display()

    # 2. Restaurant Dataset - Classic AI textbook example
    print("\n2. Restaurant Dataset Classification")
    print(f"Dataset info: {restaurant}")

    restaurant_tree = DecisionTreeLearner(restaurant)
    restaurant_error = err_ratio(restaurant_tree, restaurant)
    print(f"Training error rate: {restaurant_error:.3f}")
    print(f"Training accuracy: {(1 - restaurant_error) * 100:.1f}%")

    # Test individual predictions
    print("\nTesting individual restaurant scenarios:")
    test_examples = restaurant.examples[:3]
    for i, example in enumerate(test_examples):
        prediction = restaurant_tree(example)
        actual = example[restaurant.target]
        print(f"Example {i + 1}: Predicted={prediction}, Actual={actual}")

    # 3. Synthetic Restaurant Data - Test learning capability
    print("\n3. Synthetic Restaurant Data (Testing Learning)")
    synthetic_data = SyntheticRestaurant(50)  # Generate 50 examples
    print(f"Generated synthetic dataset: {synthetic_data}")

    synthetic_tree = DecisionTreeLearner(synthetic_data)
    synthetic_error = err_ratio(synthetic_tree, synthetic_data)
    print(f"Synthetic data training error: {synthetic_error:.3f}")
    print("(Low error suggests the algorithm can recover the original pattern)")


def demonstrate_knn():
    """Demonstrate k-Nearest Neighbors on datasets with numerical features."""
    print_separator("k-NEAREST NEIGHBORS (kNN)")

    # Iris dataset is perfect for kNN (numerical features)
    print("Iris Dataset Classification with kNN")
    print(f"Dataset info: {iris}")
    print(f"Features: {[iris.attr_names[i] for i in iris.inputs]}")
    print(f"Classes: {iris.values[iris.target]}")

    # Test different k values
    k_values = [1, 3, 5, 7]
    print(f"\nTesting different k values: {k_values}")

    for k in k_values:
        # Use train-test split for honest evaluation
        train_data, test_data = train_test_split(iris, test_split=0.3)

        # Create training dataset
        train_dataset = DataSet(
            examples=train_data, attr_names=iris.attr_names, target=iris.target
        )

        # Train new kNN on training data
        knn_train_classifier = NearestNeighborLearner(train_dataset, k=k)

        # Test on held-out data
        test_error = err_ratio(knn_train_classifier, train_dataset, test_data)
        print(f"k={k}: Test accuracy = {(1 - test_error) * 100:.1f}%")

    # Demonstrate prediction on individual examples
    print("\nIndividual predictions (k=3):")
    knn_3 = NearestNeighborLearner(iris, k=3)
    test_examples = iris.examples[:5]

    for i, example in enumerate(test_examples):
        prediction = knn_3(example)
        actual = example[iris.target]
        features = [example[j] for j in iris.inputs]
        print(
            f"Example {i + 1}: Features={features}, Predicted={prediction}, Actual={actual}"
        )


def demonstrate_linear_classifiers():
    """Demonstrate linear and logistic regression."""
    print_separator("LINEAR CLASSIFIERS")

    # For linear classifiers, we need to prepare the iris dataset with numeric targets
    print("Iris Dataset with Linear Classifiers")

    # Create a copy and convert classes to numbers
    iris_numeric = DataSet(
        examples=[ex[:] for ex in iris.examples],  # Deep copy
        attr_names=iris.attr_names,
        target=iris.target,
    )
    iris_numeric.classes_to_numbers()

    print(f"Converted classes to numbers: {iris_numeric.values[iris_numeric.target]}")

    # 1. Linear Regression
    print("\n1. Linear Regression")
    linear_model = LinearLearner(iris_numeric, learning_rate=0.01, epochs=200)

    # Test on training data
    predictions = []
    actuals = []
    for example in iris_numeric.examples:
        pred = linear_model(example)
        actual = example[iris_numeric.target]
        predictions.append(pred)
        actuals.append(actual)

    # Convert to numpy arrays for metrics
    pred_array = np.array(predictions)
    actual_array = np.array(actuals)

    # Calculate R² score for regression
    r2 = r2_score(pred_array, actual_array)
    print(f"R² Score: {r2:.3f}")

    # For classification, round predictions and calculate accuracy
    pred_rounded = np.round(pred_array).astype(int)
    # Ensure predictions are in valid range
    pred_rounded = np.clip(pred_rounded, 0, 2)
    accuracy = accuracy_score(pred_rounded, actual_array.astype(int))
    print(f"Classification Accuracy (rounded): {accuracy * 100:.1f}%")

    # 2. Logistic Regression
    print("\n2. Logistic Regression")
    logistic_model = LogisticLinearLearner(iris_numeric, learning_rate=0.1, epochs=200)

    # Test logistic regression
    log_predictions = []
    for example in iris_numeric.examples:
        pred = logistic_model(example)
        log_predictions.append(pred)

    log_pred_array = np.array(log_predictions)
    print(
        f"Logistic predictions range: [{log_pred_array.min():.3f}, {log_pred_array.max():.3f}]"
    )
    print("(Values between 0 and 1 represent probabilities)")

    # Show some individual predictions
    print("\nSample predictions (first 5 examples):")
    for i in range(5):
        linear_pred = predictions[i]
        logistic_pred = log_predictions[i]
        actual = actuals[i]
        print(
            f"Example {i + 1}: Linear={linear_pred:.2f}, Logistic={logistic_pred:.3f}, Actual={actual}"
        )


def demonstrate_cross_validation():
    """Demonstrate cross-validation techniques."""
    print_separator("CROSS-VALIDATION TECHNIQUES")

    print("Cross-validation on Zoo Dataset with Decision Trees")

    # 1. Standard k-fold cross-validation
    print("\n1. 5-Fold Cross-Validation")
    train_err, val_err = cross_validation(DecisionTreeLearner, zoo, k=5)
    print(f"Average training error: {train_err:.3f}")
    print(f"Average validation error: {val_err:.3f}")
    print(f"Average validation accuracy: {(1 - val_err) * 100:.1f}%")

    # 2. Leave-one-out cross-validation
    print("\n2. Leave-One-Out Cross-Validation")
    loo_train_err, loo_val_err = leave_one_out(DecisionTreeLearner, zoo)
    print(f"LOO training error: {loo_train_err:.3f}")
    print(f"LOO validation error: {loo_val_err:.3f}")
    print(f"LOO validation accuracy: {(1 - loo_val_err) * 100:.1f}%")

    # 3. Cross-validation with different algorithms
    print("\n3. Comparing Algorithms with Cross-Validation")
    algorithms = [
        ("Decision Tree", DecisionTreeLearner),
        ("1-NN", lambda dataset: NearestNeighborLearner(dataset, k=1)),
        ("3-NN", lambda dataset: NearestNeighborLearner(dataset, k=3)),
        ("5-NN", lambda dataset: NearestNeighborLearner(dataset, k=5)),
    ]

    for name, algorithm in algorithms:
        train_err, val_err = cross_validation(algorithm, zoo, k=5)
        print(f"{name:12}: Validation accuracy = {(1 - val_err) * 100:.1f}%")


def demonstrate_performance_metrics():
    """Demonstrate various performance evaluation metrics."""
    print_separator("PERFORMANCE EVALUATION METRICS")

    # Create some example predictions for demonstration
    print("Performance Metrics Demonstration")

    # Classification metrics
    print("\n1. Classification Metrics")
    y_true_class = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    y_pred_class = np.array([0, 1, 1, 1, 0, 2, 0, 0])

    accuracy = accuracy_score(y_pred_class, y_true_class)
    print(f"True labels:      {y_true_class}")
    print(f"Predicted labels: {y_pred_class}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")

    # Regression metrics
    print("\n2. Regression Metrics")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 2.2, 2.8, 3.9, 5.1])

    r2 = r2_score(y_pred_reg, y_true_reg)
    print(f"True values:      {y_true_reg}")
    print(f"Predicted values: {y_pred_reg}")
    print(f"R² Score: {r2:.3f}")
    print("(R² = 1.0 is perfect, R² = 0.0 means no better than predicting the mean)")


def demonstrate_data_handling():
    """Demonstrate dataset manipulation and utilities."""
    print_separator("DATA HANDLING AND UTILITIES")

    print("Dataset Information and Manipulation")

    # 1. Dataset properties
    datasets = [
        ("Iris", iris),
        ("Zoo", zoo),
        ("Restaurant", restaurant),
        ("O-rings", orings),
    ]

    print("\n1. Dataset Overview")
    for name, dataset in datasets:
        print(
            f"{name:10}: {len(dataset.examples)} examples, {len(dataset.attrs)} attributes"
        )
        print(f"{'':10}  Target: {dataset.attr_names[dataset.target]}")
        print(f"{'':10}  Features: {len(dataset.inputs)} input attributes")

    # 2. Train-test splitting
    print("\n2. Train-Test Split Demonstration")
    train_data, test_data = train_test_split(iris, test_split=0.2)
    print(f"Original dataset: {len(iris.examples)} examples")
    print(f"Training set: {len(train_data)} examples (80%)")
    print(f"Test set: {len(test_data)} examples (20%)")

    # 3. Class distribution
    print("\n3. Class Distribution Analysis")
    class_counts = {}
    for example in iris.examples:
        class_label = example[iris.target]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

    print("Iris dataset class distribution:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count} examples")


if __name__ == "__main__":
    """Main function to run all demonstrations."""
    print("MACHINE LEARNING TECHNIQUES DEMONSTRATION")
    print("=========================================")
    print("This program demonstrates various ML techniques from learning.py")

    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Run all demonstrations
    demonstrate_decision_trees()
    demonstrate_knn()
    demonstrate_linear_classifiers()
    demonstrate_cross_validation()
    demonstrate_performance_metrics()
    demonstrate_data_handling()

    print_separator("DEMONSTRATION COMPLETE")
