import numpy as np
import pandas as pd
import math

class Node:
    def __init__(self, feature=None, threshold=None, edges=None, value=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.edges = edges or []
        self.value = value
        self.is_leaf = is_leaf

def info_gain(df, features, label):
    feat_to_gain = {}
    for f in features:
        gain = 0
        feat = df[f]
        values = df[f].unique()
        if pd.api.types.is_numeric_dtype(feat):
            values = ['<=', '>']
            avg = df[f].mean()
            # replace the numbers so we don't try and calculate the entropy of every possible split, 
            # we just split on the average value for numeric features
            feat = df[f].apply(lambda x: '<=' if x <= avg else '>')
        for val in values:
            entropy = 0
            subset = df[feat == val]
            # avoid divide by 0 error--we just won't consider this value in our entropy
            if len(subset) == 0:
                continue
            for cls in df[label].unique():
                prob = len(subset[subset[label] == cls]) / len(subset)
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            gain += entropy
            feat_to_gain[f] = gain
    return feat_to_gain

def make_dt(df, features_left, max_depth, label, current_depth=0):
    node = Node()
    if len(df[label].unique()) == 1:
        node.is_leaf = True
        node.value = df[label].iloc[0]
        return node
    if len(features_left) == 0 or current_depth >= max_depth:
        node.is_leaf = True
        node.value = df[label].mode()[0]
        return node
    best_split_feature = max(info_gain(df, features_left, label), key=info_gain(df, features_left, label).get)
    node.feature = best_split_feature
    features_left.remove(best_split_feature)

    # numeric case
    if pd.api.types.is_numeric_dtype(df[best_split_feature]):
        avg = df[best_split_feature].mean()
        partition_leq = df[df[best_split_feature] <= avg]
        partition_gt = df[df[best_split_feature] > avg]
        node.threshold = avg
        if len(partition_leq) == 0:
            left_node = Node(is_leaf=True, value=df[label].mode()[0])
        else:
            left_node = make_dt(partition_leq, [*features_left], max_depth, label, current_depth+1)
        if len(partition_gt) == 0:
            right_node = Node(is_leaf=True, value=df[label].mode()[0])
        else:
            right_node = make_dt(partition_gt, [*features_left], max_depth, label, current_depth+1)
        node.edges.append(('<=', left_node))
        node.edges.append(('>', right_node))

    # categorical case
    else: 
        values = df[best_split_feature].unique()
        for v in values:
            partition_v = df[df[best_split_feature] == v]
            if len(partition_v) == 0:
                leaf = Node(is_leaf=True, value=df[label].mode()[0])
            else:
                leaf = make_dt(partition_v, [*features_left], max_depth, label, current_depth+1)
            node.edges.append((v, leaf))
    return node

def traverse(node, sample):
    if node.is_leaf:
        return node.value
    # numeric case
    val = sample[node.feature]
    if node.threshold is not None:
        if val <= node.threshold:
            val = '<='
        else:
            val = '>'
    for edge in node.edges:
        if val == edge[0]:
            return traverse(edge[1], sample)

def bootstrap_data(X, n_samples):
    indices = np.random.choice(X.index, size=n_samples, replace=True)
    return X.loc[indices]

def choose_random_features(features, m):
    return list(np.random.choice(features, size=m, replace=False))

def random_forest(df, n_trees, max_depth, label):
    # make n_trees
    trees = []
    features = list(df.columns)
    m = int(math.sqrt(len(features)-1))
    features.remove(label)
    for _ in range(n_trees):
        bootstrapped_df = bootstrap_data(df, len(df))
        random_features = choose_random_features(features, m)
        # make decision tree with bootstrapped data and random features
        tree = make_dt(bootstrapped_df, random_features, max_depth, label)
        trees.append(tree)
    return trees

# gets the majority vote from the trees
def traverse_forest(trees, sample):
    votes = []
    for tree in trees:
        votes.append(traverse(tree, sample))
    return max(set(votes), key=votes.count)

def make_predictions(forest, test, label):
    predictions = test
    predictions[label] = predictions.apply(lambda x: traverse_forest(forest, x), axis=1)
    return predictions

def compute_metrics(predictions, truth, label):
    tp = np.sum((predictions[label] == 1) & (truth[label] == 1))
    tn = np.sum((predictions[label] == 0) & (truth[label] == 0))
    fp = np.sum((predictions[label] == 1) & (truth[label] == 0))
    fn = np.sum((predictions[label] == 0) & (truth[label] == 1))
    accuracy = (tp + tn) / (tp+tn+fp+fn)
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    
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

def kfold(df, k, ntree, max_depth, label):
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
        forest = random_forest(train, ntree, max_depth, label)
        predictions = make_predictions(forest, test, label)
        acc, pre, rec, f1 = compute_metrics(predictions, truth, label)
        accuracy += acc
        precision += pre
        recall += rec
        f1_score += f1
    return [accuracy / k, precision / k, recall / k, f1_score / k]

def kfold_classes(df, k, ntree, max_depth, pos_class, neg_class, label):
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
        # make predictions 
        forest = random_forest(train, ntree, max_depth, label)
        predictions = make_predictions(forest, test, label)
        acc, pre, rec, f1 = compute_metrics_classes(predictions, truth, pos_class, neg_class, label)
        accuracy += acc
        precision += pre
        recall += rec
        f1_score += f1
    return [accuracy / k, precision / k, recall / k, f1_score / k]


def evaluate_forest(df, ntrees, label, max_depth):
    ntree_to_metrics = dict.fromkeys(ntrees)
    for ntree in ntrees:
        [acc, pre, rec, f1] = kfold(df, 10, ntree, max_depth, label)
        ntree_to_metrics[ntree] = (acc, pre, rec, f1)
        print(f"Finished {ntree} trees")
    return ntree_to_metrics

def evaluate_forest_classes(df, ntrees, label, pos_class, neg_class, max_depth):
    ntree_to_metrics = dict.fromkeys(ntrees)
    for ntree in ntrees:
        [acc, pre, rec, f1] = kfold_classes(df, 10, ntree, max_depth, pos_class, neg_class, label)
        ntree_to_metrics[ntree] = (acc, pre, rec, f1)
        print(f"Finished {ntree} trees")
    return ntree_to_metrics