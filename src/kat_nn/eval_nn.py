import numpy as np
import pandas as pd
from neural_network import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def pred(X, Thetas, activation):
    """Helper function to calculate model predictions."""
    A, _ = forwardprop(X, Thetas, activation)
    return A[-1]

def predict(X, Thetas, activation, threshold=0.5):
    """Helper function to determine model classifications."""
    probs = pred(X, Thetas, activation)

    if probs.ndim == 1 or probs.shape[0] == 1:
        probs = probs.ravel()
        return (probs >= threshold).astype(int)

    return np.argmax(probs, axis=0)

def evaluate_model(X, y, Thetas, activation, threshold=0.5):
    """Helper function to calculate accuracy and f1-value of model."""
    y_true = np.asarray(y).ravel().astype(int)
    y_pred = predict(X, Thetas, activation, threshold=threshold)

    if len(np.unique(y_true)) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
    else:
        accuracy = np.mean(y_true == y_pred)
        classes = np.unique(y_true)
        f1s = []

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1_cls = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            f1s.append(f1_cls)

        f1 = np.mean(f1s)

    return accuracy, f1

def normalize_train_test(X_train, X_test):
    """Helper function to normalize train and test sets."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm

def make_k_stratified_folds(D, k):
    """Helper function to make k stratified folds."""
    y = D[:, -1]
    classes = np.unique(y)

    class_folds = {}

    for cls in classes:
        class_rows = D[y == cls]
        shuffled = class_rows.copy()
        np.random.default_rng().shuffle(shuffled)
        class_folds[cls] = np.array_split(shuffled, k)

    folds = []

    for i in range(k):
        fold = np.concatenate([class_folds[cls][i] for cls in classes], axis=0)
        np.random.default_rng().shuffle(fold)
        folds.append(fold)

    return folds


def stratified_k_fold_validation(D, architectures, lambdas, k, alphas, activation=sigmoid, cost_fn=cost, num_epochs=1000, batch_size=None):
    """Returns the results of running k-fold validation neural network"""
    
    performances = []
    num_features = D.shape[1]-1

    folds = make_k_stratified_folds(D, k)

    for layers in architectures:
        num_classes = len(np.unique(D[:, -1]))

        if num_classes == 2:
            output_size = 1
        else:
            output_size = num_classes

        layer_sizes = [num_features] + list(layers) + [output_size]

        for lambda_reg in lambdas:
            for alpha in alphas:
                fold_accs = []
                fold_f1s = []

                for i in range(k):
                    train = np.empty((0, num_features+1))
                    for j in range(k):
                        if j != i:
                            train = np.concatenate((train, folds[j]))

                    X_train = train[:, :-1]
                    y_train = train[:, -1]
                    X_test = folds[i][:, :-1]
                    y_test = folds[i][:, -1]

                    X_train, X_test = normalize_train_test(X_train, X_test)

                    if output_size > 1:
                        y_train_encoded = np.eye(output_size)[y_train.astype(int)].T
                    else:
                        y_train_encoded = y_train

                    Thetas, _ = nn(
                        X_train,
                        y_train_encoded,
                        layer_sizes=layer_sizes,
                        activation=activation,
                        cost_fn=cost_fn,
                        alpha=alpha,
                        lambda_reg=lambda_reg,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=False
                    )
                    
                    acc, f1 = evaluate_model(X_test, y_test, Thetas, activation)
                    fold_accs.append(acc)
                    fold_f1s.append(f1)

                performances.append({
                    "architecture": str(layers),
                    "layer_sizes": str(layer_sizes),
                    "lambda": lambda_reg,
                    "alpha": alpha,
                    "accuracy_mean": np.mean(fold_accs),
                    "accuracy_std": np.std(fold_accs),
                    "f1_mean": np.mean(fold_f1s),
                    "f1_std": np.std(fold_f1s),
                })

    results_df = pd.DataFrame(performances)
    results_df = results_df.sort_values(by=["f1_mean", "accuracy_mean"], ascending=False).reset_index(drop=True)
    return results_df

def best_configuration(results_df, metric="f1_mean"):
    idx = results_df[metric].idxmax()
    return results_df.loc[idx]

def learning_curve_cost(D, hidden_layers, lambda_reg, alpha, train_sizes, cost_fn=cost, activation=sigmoid, num_epochs=1000, batch_size=None):
    """Returns training sizes and test-set costs."""
    X = D[:, :-1]
    y = D[:, -1]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    X_train_full_normalized, X_test_normalized = normalize_train_test(X_train_full, X_test)

    test_costs = []
    used_sizes = []

    num_features = X.shape[1]
    num_classes = len(np.unique(D[:, -1]))

    if num_classes == 2:
        output_size = 1
    else:
        output_size = num_classes

    layer_sizes = [num_features] + list(hidden_layers) + [output_size]

    for n_train in train_sizes:
        if n_train > len(X_train_full):
            continue

        X_sub_raw, _, y_sub, _ = train_test_split(
            X_train_full_normalized, y_train_full,
            train_size=n_train,
            stratify=y_train_full
        )

        if output_size > 1:
            y_sub_encoded = np.eye(output_size)[y_sub.astype(int)].T
            y_test_encoded = np.eye(output_size)[y_test.astype(int)].T
        else:
            y_sub_encoded = y_sub.reshape(1, -1)
            y_test_encoded = y_test.reshape(1, -1)

        Thetas, _ = nn(
            X_sub_raw,
            y_sub_encoded,
            layer_sizes=layer_sizes,
            activation=activation,
            cost_fn=cost_fn,
            alpha=alpha,
            lambda_reg=lambda_reg,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=False
        )

        A, _ = forwardprop(X_test_normalized, Thetas, activation)
        AL = A[-1]

        J_test = cost_fn(y_test_encoded, AL, Thetas, lambda_reg)

        used_sizes.append(n_train)
        test_costs.append(J_test)

    return used_sizes, test_costs

def plot_learning_curve(train_sizes, costs, dataset_name, title="Learning Curve"):
    plt.figure(figsize=(7, 4))
    plt.plot(train_sizes, costs, marker="o")
    plt.xlabel("Number of training samples")
    plt.ylabel("Cost J")
    plt.title(title)
    plt.grid(True)
    plt.savefig("2tmp/" + dataset_name + "_learning_curve" + ".jpg")