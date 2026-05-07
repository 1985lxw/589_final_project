import pandas as pd
import numpy as np

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
        # make predictions 
        model = knn(train, train[label], neighbors)
        predictions = make_predictions(model, test, label)
        acc, pre, rec, f1 = compute_metrics(predictions, truth, label)
        accuracy += acc
        precision += pre
        recall += rec
        f1_score += f1
    return [accuracy / k, precision / k, recall / k, f1_score / k]