import numpy as np
import pandas as pd
import random
import math

class NN:
    def __init__(self, weights, l, alpha, epochs=10, verbose=False):
        self.weights = weights
        self.l = l
        self.alpha = alpha
        self.epochs = epochs
        self.verbose = verbose

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def g(self, a):
        return 1/(1 + np.exp(-a))

    def loss(self, predicted, y, reg_term=0):
        # no divide be 0 error pls
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -np.sum(y*np.log(predicted)+(1-y)*np.log(1-predicted)) + reg_term
    
    def forward(self, x):
        # prepare network input and add bias term
        x = np.atleast_1d(x).flatten()
        a = [np.concatenate(([1], x), axis=0)]
        self.log(f"a1: {a}")
        L = len(self.weights)
        for k in range(L):
            z_k = self.weights[k] @ a[-1]
            self.log(f"z{k+2} {z_k}")
            a_k = self.g(z_k) # activation output of all neurons in the k-th hidden layer
            if k < L - 1:
                a_k = np.concatenate(([1], a_k), axis=0)
            self.log(f"a{k+2} {a_k}")
            a.append(a_k)
        return a

    def backprop(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        D = [np.zeros(w.shape) for w in self.weights]
        num_layers = len(self.weights)
        i = 1
        predicted = np.array([])
        total_cost = 0
        for x_i, y_i in zip(X, y):
            # 1.1 propagate x_i
            a = self.forward(x_i)
            self.log(f"\nProcessing training instance {i}: ")
            # 1.2
            np.append(predicted, a[-1])
            self.log(f"Predicted: {a[-1]}, Actual: {y_i}")
            instance_cost = self.loss(a[-1], y_i)
            total_cost += instance_cost
            self.log(f"Cost: {instance_cost}")
            self.log(f"\nBackpropagation values of instance {i}: ")
            deltas = [None for _ in range(len(a))]
            deltas[-1] = a[-1] - y_i # computes delta values of all neurons in the output layers
            # 1.3 compute delta values of all neurons in the hidden layers
            for k in range(num_layers-1, 0, -1):
                delta_k = np.multiply(self.weights[k].T @ deltas[k+1], np.multiply(a[k], 1 - a[k]))
                # remove delta associated with the bias neuron of k-th layer of network
                delta_k = delta_k[1:]
                deltas[k] = delta_k
            self.log("Deltas: ", [(f"Delta {k+1}", deltas[k]) for k in range(len(deltas)) if deltas[k] is not None])
            # 1.4 updates the gradients of the weights of each layer based on current training instance
            for k in range(num_layers):
                D[k] += np.outer(deltas[k+1], a[k])
                self.log(f"Gradients of Theta {k+1}", np.outer(deltas[k+1], a[k]))
            i += 1
        # 2. compute the final regularized gradients of the weights of each layer
        for k in range(num_layers):
            # 2.1 
            P = self.l * self.weights[k]
            # set first column to be all 0 for the bias weight
            P[:,0] = 0
            # 2.2 combines gradients with regularization terms and divides by num instances
            # to obtain the average gradient
            D[k] = (D[k] + P)/len(X)
        # 4. update weights of each layer based on corresponding gradients
        for k in range(num_layers):
            self.weights[k] -= self.alpha * D[k]
        self.log(f"Final Cost: ", total_cost / len(X))
        self.log("\nFinal gradients:", [(f"Theta {k+1}", D[k]) for k in range(len(D))])
        return self.weights

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        for i in range(self.epochs):
            weights = self.backprop(X, y)
            self.weights = weights
        return weights
    
    def batch(self, X, y, batch, epochs):
        X = X.to_numpy()
        y = y.to_numpy()
        for _ in range(epochs):
            idxs = random.sample(range(len(X)), batch)
            weights = self.backprop([X[i] for i in idxs], [y[i] for i in idxs])
        return weights
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = []
        for x in X:
            a = self.forward(x)
            predictions.append(1 if a[-1][0] >= 0.5 else 0)
        return pd.DataFrame(predictions, columns=['label'])

# outside of the class
def create_weights(layers):
    weights = [] 
    for i in range(len(layers)-1): 
        last = layers[i]
        curr = layers[i+1]
        # randomly sample weights from a normal distribution 
        weights.append(np.random.normal(0, 1, (curr, last + 1)).astype(np.longdouble))
    return weights

def norm_col(col):
    # using min-max
    mi = min(col)
    ma = max(col)
    return col.map(lambda a: (a-mi)/(ma - mi))

def normalize(X):
    return X.apply(norm_col)

def compute_metrics(predictions, truth):
        tp = np.sum((predictions['label'] == 1) & (truth['label'] == 1))
        tn = np.sum((predictions['label'] == 0) & (truth['label'] == 0))
        fp = np.sum((predictions['label'] == 1) & (truth['label'] == 0))
        fn = np.sum((predictions['label'] == 0) & (truth['label'] == 1))
        
        # print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        # print(f"Sum: {tp + tn + fp + fn}, Expected: {len(predictions)}")
        # print(f"Predictions value counts: {predictions['label'].value_counts().to_dict()}")
        
        accuracy = (tp + tn) / (tp+tn+fp+fn)
        precision = tp / (tp + fp) 
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return accuracy, precision, recall, f1

def kfold(df, k, nn):
    fold_size = len(df) // k
    # split data into label classes to preserve class distribution in folds
    class_0_prob = len(df[df['label'] == 0]) / len(df)
    class_1_prob = len(df[df['label'] == 1]) / len(df)
    folds = [] # holds indices for all folds
    sample_df = df.copy()
    for _ in range(k):
        # sample from each class according to their distribution
        class_0_sample = sample_df[sample_df['label'] == 0].sample(n=math.floor(class_0_prob*fold_size)).index
        class_1_sample = sample_df[sample_df['label'] == 1].sample(n=math.floor(class_1_prob*fold_size)).index
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
        test = truth.drop(columns=['label'])
        # fit the model
        y = train['label']
        X = train.drop(columns=['label'])
        nn.fit(X, y)
        # make predictions
        predictions = nn.predict(test)
        # to avoid error
        predictions.reset_index(drop=True, inplace=True)
        truth = truth.reset_index(drop=True)
        acc, pre, rec, f1 = compute_metrics(predictions, truth)
        accuracy += acc
        precision += pre
        recall += rec
        f1_score += f1
    return [accuracy / k, precision / k, recall / k, f1_score / k]