import numpy as np

class Node:
    def __init__(self, feature=None, edges=None, value=None, is_leaf=False):
        self.feature = feature
        self.edges = edges or []
        self.value = value
        self.is_leaf = is_leaf

def info_gain(df, features):
    feat_to_gain = {}
    for f in features:
        gain = 0
        for val in df[f].unique():
            entropy = 0
            subset = df[df[f] == val]
            for cls in df['class'].unique():
                prob = len(subset[subset['class'] == cls]) / len(df)
                if prob > 0:
                    entropy -= prob * np.log2(prob)
        gain += entropy
        feat_to_gain[f] = gain
    return feat_to_gain

def make_dt(df, features_left):
    node = Node()
    if len(df['class'].unique()) == 1:
        node.is_leaf = True
        node.value = df['class'].iloc[0]
        return node
    if len(features_left) == 0:
        node.is_leaf = True
        node.value = df['class'].mode()[0]
        return node
    best_split_feature = max(info_gain(features_left), key=info_gain(features_left).get)
    node.feature = best_split_feature
    features_left.remove(best_split_feature)

    values = df[best_split_feature].unique()
    for v in values:
        partition_v = df[df[best_split_feature] == v]
        if len(partition_v) == 0:
            leaf = Node(is_leaf=True, value=df['class'].mode()[0])
        else:
            leaf = make_dt(partition_v, [*features_left])
        node.edges.append((v, leaf))
    return node

def traverse(node, sample):
    if node.is_leaf:
        return node.value
    for edge in node.edges:
        if sample[node.feature] == edge[0]:
            return traverse(edge[1], sample)

def make_predictions(tree, df):
    predictions = df
    predictions['class'] = predictions.apply(lambda x: traverse(tree, x), axis=1)
    return predictions