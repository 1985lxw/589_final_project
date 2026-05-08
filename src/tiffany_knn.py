import pandas as pd
import numpy as np
import math

def knn(X_train, y_train, sample, k):
    # keep track of the distance between every instance in the training set and the test sample
    dist_df = pd.DataFrame(columns=X_train.columns)
    for col in X_train.columns:
        dist_df[col] = (X_train[col] - sample[col]) ** 2
    dist_df["distance"] = np.sqrt(dist_df.sum(axis=1))
    # then sort the distances and return the indices of the k nearest neighbors
    neighbors = dist_df.sort_values(by='distance').head(k).index
    # then return the most common class among the neighbors
    majority_class = y_train[neighbors].mode()[0]
    return majority_class

def make_predictions(X_train, y_train, test, k, label):
    predictions = test.copy()
    predictions[label] = predictions.apply(lambda x: knn(X_train, y_train, x, k), axis=1)
    return predictions

def compute_metrics(predictions, truth, label):
    tp = np.sum((predictions[label] == 1) & (truth[label] == 1))
    tn = np.sum((predictions[label] == 0) & (truth[label] == 0))
    fp = np.sum((predictions[label] == 1) & (truth[label] == 0))
    fn = np.sum((predictions[label] == 0) & (truth[label] == 1))
    accuracy = (tp + tn) / (tp+tn+fp+fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def compute_metrics_classes(predictions, truth, pos_class, neg_class, label):
    tp = np.sum((predictions[label] == pos_class) & (truth[label] == pos_class))
    tn = np.sum((predictions[label] == neg_class) & (truth[label] == neg_class))
    fp = np.sum((predictions[label] == pos_class) & (truth[label] == neg_class))
    fn = np.sum((predictions[label] == neg_class) & (truth[label] == pos_class))
    accuracy = (tp + tn) / (tp+tn+fp+fn)
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1

def compute_metrics_numbers(predictions, truth, label):
    accuracy = np.sum(predictions[label] == truth[label]) / len(truth)
    classes = np.unique(truth[label])
    precision_list = []
    recall_list = []
    f1_list = []
    
    for cls in classes:
        tp = np.sum((predictions[label] == cls) & (truth[label] == cls))
        fp = np.sum((predictions[label] == cls) & (truth[label] != cls))
        fn = np.sum((predictions[label] != cls) & (truth[label] == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    
    return accuracy, precision, recall, f1

def kfold(df, k, neighbors, label):
    fold_size = len(df) // k
    # split data into label classes to preserve class distribution in folds
    class_0_prob = len(df[df[label] == 0]) / len(df)
    class_1_prob = len(df[df[label] == 1]) / len(df)
    folds = [] # holds indices for all folds
    sample_df = df.copy()
    for _ in range(k):
        # sample from each class according to their distribution
        class_0_sample = sample_df[sample_df[label] == 0].sample(n=math.floor(class_0_prob*fold_size)).index
        class_1_sample = sample_df[sample_df[label] == 1].sample(n=math.floor(class_1_prob*fold_size)).index
        # remove samples from df
        sample_df = sample_df.drop(class_0_sample)
        sample_df = sample_df.drop(class_1_sample)
        # then add the samples to the folds list
        fold_indices = class_0_sample.append(class_1_sample)
        folds.append(fold_indices)

    # train and test on folds
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    for fold in folds:
        # gets datapoints for train and test sets from folds
        # current fold is test set, rest are train set
        train = df.drop(index=fold)
        truth = df.loc[fold]
        test = truth.drop(columns=[label])
        # make predictions using knn
        predictions = make_predictions(train.drop(columns=[label]), train[label], test, neighbors, label)
        acc, pre, rec, f1 = compute_metrics(predictions, truth, label)
        accuracy += acc
        f1_score += f1
    return [accuracy / k, f1_score / k]

def kfold_classes(df, k, neighbors, pos_class, neg_class,label):
    fold_size = len(df) // k
    # split data into label classes to preserve class distribution in folds
    class_0_prob = len(df[df[label] == neg_class]) / len(df)
    class_1_prob = len(df[df[label] == pos_class]) / len(df)
    folds = [] # holds indices for all folds
    sample_df = df.copy()
    for _ in range(k):
        # sample from each class according to their distribution
        class_0_sample = sample_df[sample_df[label] == neg_class].sample(n=math.floor(class_0_prob*fold_size)).index
        class_1_sample = sample_df[sample_df[label] == pos_class].sample(n=math.floor(class_1_prob*fold_size)).index
        # remove samples from df
        sample_df = sample_df.drop(class_0_sample)
        sample_df = sample_df.drop(class_1_sample)
        # then add the samples to the folds list
        fold_indices = class_0_sample.append(class_1_sample)
        folds.append(fold_indices)

    # train and test on folds
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    for fold in folds:
        # gets datapoints for train and test sets from folds
        # current fold is test set, rest are train set
        train = df.drop(index=fold)
        truth = df.loc[fold]
        test = truth.drop(columns=[label])
        # make predictions using knn
        predictions = make_predictions(train.drop(columns=[label]), train[label], test, neighbors, label)
        acc, pre, rec, f1 = compute_metrics_classes(predictions, truth, pos_class, neg_class, label)
        accuracy += acc
        f1_score += f1
    return [accuracy / k, f1_score / k]

def kfold_numbers(df, k, neighbors, label):
    # Get all unique classes from the data
    classes = df[label].unique()
    fold_size = len(df) // k
    
    # Calculate class probabilities
    class_probs = {}
    for cls in classes:
        class_probs[cls] = len(df[df[label] == cls]) / len(df)
    
    # Create folds stratified by all classes
    folds = []
    sample_df = df.copy()
    for _ in range(k):
        fold_indices = None
        for cls in classes:
            cls_sample = sample_df[sample_df[label] == cls].sample(n=math.floor(class_probs[cls]*fold_size)).index
            if fold_indices is None:
                fold_indices = cls_sample
            else:
                fold_indices = fold_indices.append(cls_sample)
        # remove samples from df
        sample_df = sample_df.drop(fold_indices)
        folds.append(fold_indices)

    # train and test on folds
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    for fold in folds:
        # gets datapoints for train and test sets from folds
        # current fold is test set, rest are train set
        train = df.drop(index=fold)
        truth = df.loc[fold]
        test = truth.drop(columns=[label])
        # make predictions using knn
        predictions = make_predictions(train.drop(columns=[label]), train[label], test, neighbors, label)
        acc, pre, rec, f1 = compute_metrics_numbers(predictions, truth, label)
        accuracy += acc
        precision += pre
        recall += rec
        f1_score += f1
    return [accuracy / k, precision / k, recall / k, f1_score / k]

def norm_col(col):
    # using min-max
    mi = min(col)
    ma = max(col)
    return col.map(lambda a: (a-mi)/(ma - mi))

def normalize(X):
    return X.apply(norm_col)