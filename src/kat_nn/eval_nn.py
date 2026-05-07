import numpy as np
import pandas as pd
from neural_network import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def pred(X, Thetas):
    """Helper function to calculate model predictions."""
    A, _ = forwardprop(X, Thetas)
    AL = A[-1] 
    return AL.ravel()

def predict(X, Thetas, threshold=0.5):
    """Helper function to determine model classifications."""
    probs = pred(X, Thetas)
    return (probs >= threshold).astype(int)

def evaluate_model(X, y, Thetas):
    """Helper function to calculate accuracy and f1 of the model."""
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    y_true = np.asarray(y).ravel()
    y_pred = predict(X, Thetas)

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                true_pos += 1
            else:
                false_neg += 1
        elif y_true[i] == 0:
            if y_pred[i] == 0:
                true_neg +=1 
            else:
                false_pos += 1


    accuracy = (true_pos + true_neg) / (len(y_pred))
    precision = true_pos / (true_pos + false_pos) 
    recall = true_pos / (true_pos + false_neg) 
    f1 = (2 * precision * recall) / (precision + recall) 
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
    D_0 = D[D[:, -1] == 0]
    D_1 = D[D[:, -1] == 1]

    np.random.default_rng().shuffle(D_0)
    np.random.default_rng().shuffle(D_1)

    folds_0 = np.array_split(D_0, k)
    folds_1 = np.array_split(D_1, k)

    folds = []

    for i in range(k):
        fold = np.concatenate([folds_0[i], folds_1[i]], axis=0)
        folds.append(fold)

    return folds


def stratified_k_fold_validation(D, architectures, lambdas, k, alphas, num_epochs=1000, batch_size=None):
    """Returns the results of running k-fold validation neural network"""
    
    performances = []
    num_features = D.shape[1]-1

    folds = make_k_stratified_folds(D, k)

    for layers in architectures:
        layer_sizes = [num_features] + list(layers) + [1]

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

                    Thetas, _ = nn(
                        X_train,
                        y_train,
                        layer_sizes=layer_sizes,
                        alpha=alpha,
                        lambda_reg=lambda_reg,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=False
                    )
                    
                    acc, f1 = evaluate_model(X_test, y_test, Thetas)
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

def learning_curve_cost(D, hidden_layers, lambda_reg, alpha, train_sizes, num_epochs=1000, batch_size=None):
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
    layer_sizes = [num_features] + list(hidden_layers) + [1]

    for n_train in train_sizes:
        if n_train > len(X_train_full):
            continue

        X_sub_raw, _, y_sub, _ = train_test_split(
            X_train_full_normalized, y_train_full,
            train_size=n_train,
            stratify=y_train_full
        )

        Thetas, _ = nn(
            X_sub_raw,
            y_sub,
            layer_sizes=layer_sizes,
            alpha=alpha,
            lambda_reg=lambda_reg,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=False
        )

        A, _ = forwardprop(X_test_normalized, Thetas)
        AL = A[-1]
        y_test_2d = y_test.reshape(1, -1)

        J_test = cost(y_test_2d, AL, Thetas)

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
    plt.savefig("589_final_project/figures/" + dataset_name + "_learning_curve" + ".jpg")