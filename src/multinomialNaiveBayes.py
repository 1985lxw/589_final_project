import re
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
import math
import numpy as np   
from sklearn import utils
import matplotlib.pyplot as plt
from collections import Counter


REPLACE_NO_SPACE = re.compile("[._;:!*`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = REPLACE_NO_SPACE.sub("", text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = text.split()
    return [w for w in words if w not in stop_words]

def load_training_set(percentage_positives, percentage_negatives):
    vocab = set()
    positive_instances = []
    negative_instances = []

    df = pd.read_csv('train-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
        vocab = vocab.union(set(contents))

    df = pd.read_csv('train-negative.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)
        vocab = vocab.union(set(contents))

    return positive_instances, negative_instances, vocab


def load_test_set(percentage_positives, percentage_negatives):
    positive_instances = []
    negative_instances = []

    df = pd.read_csv('test-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
    df = pd.read_csv('test-negative.csv')
    
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)

    return positive_instances, negative_instances

# multinomial naive bayes alg
def multinomialNB(pos_train, neg_train, pos_test, neg_test, vocab):
    # create vocab list
    # vocab = set()
    # for doc in pos_train + neg_train:
    #     vocab.update(doc)

    # calculate word freq
    count_pos = Counter()
    count_neg = Counter()
    
    for doc in pos_train:
        count_pos.update(doc)

    for doc in neg_train:
        count_neg.update(doc)

    total_pos_words = sum(count_pos.values())
    total_neg_words = sum(count_neg.values())

    # calculate prior probabilities
    num_pos = len(pos_train)
    num_neg = len(neg_train)
    total_num = num_pos + num_neg
    
    prob_pos = num_pos / total_num
    prob_neg = num_neg / total_num

    # calculate conditional probabilities   
    def conditional_prob(word, counts, total_words):
        if total_words == 0:
            return 0
        return counts[word] / total_words

    # predict??
    def predict(doc):
        total_prob_pos = prob_pos
        total_prob_neg = prob_neg
        for word in doc:
            if word not in vocab:
                continue
            word_prob_pos = conditional_prob(word, count_pos, total_pos_words)
            word_prob_neg = conditional_prob(word, count_neg, total_neg_words)
            if prob_pos > 0:
                total_prob_pos *= word_prob_pos
            if prob_neg > 0:
                total_prob_neg *= word_prob_neg

        return 1 if (total_prob_pos > total_prob_neg) else 0
    
    # return TP, FP, TN, FN
    TP = FP = TN = FN = 0

    for doc in pos_test:
        pred = predict(doc)
        if pred == 1:
            TP += 1
        else:
            FN += 1

    for doc in neg_test:
        pred = predict(doc)
        if pred == 0:
            TN += 1
        else:
            FP += 1
    return TP, FP, TN, FN

def accuracy(TP, FP, TN, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def confusion_matrix(TP, FP, TN, FN):
    print()
    print("Confusion Matrix")
    print("                Predicted")
    print("              Pos      Neg")
    print(f"Actual Pos    {TP:<8}{FN}")
    print(f"Actual Neg    {FP:<8}{TN}")


def LaPlaceSmoothedMNB(pos_train, neg_train, pos_test, neg_test, vocab, alpha):
    # create vocab list
    # vocab = set()
    # for doc in pos_train + neg_train:
    #     vocab.update(doc)

    # calculate word freq
    V = len(vocab)
    count_pos = Counter()
    count_neg = Counter()
    
    for doc in pos_train:
        count_pos.update(doc)

    for doc in neg_train:
        count_neg.update(doc)

    total_pos_words = sum(count_pos.values())
    total_neg_words = sum(count_neg.values())

    # calculate prior probabilities
    num_pos = len(pos_train)
    num_neg = len(neg_train)
    total_num = num_pos + num_neg
    
    prob_pos = num_pos / total_num
    prob_neg = num_neg / total_num

    # calculate conditional probabilities with laplace smoothing   
    def conditional_prob(word, counts, total_words, alpha):
        return (counts[word] + alpha) / (total_words + alpha * V)
    # predict??
    def predict(doc, alpha):
        log_pos = math.log(prob_pos)
        log_neg = math.log(prob_neg)

        for word in doc:
            if word in vocab:
                pw_pos = conditional_prob(word, count_pos, total_pos_words, alpha)
                pw_neg = conditional_prob(word, count_neg, total_neg_words, alpha)

                log_pos += math.log(pw_pos)
                log_neg += math.log(pw_neg)

        return 1 if (log_pos > log_neg) else 0
    
    # return TP, FP, TN, FN
    TP = FP = TN = FN = 0

    for doc in pos_test:
        pred = predict(doc, alpha)
        if pred == 1:
            TP += 1
        else:
            FN += 1

    for doc in neg_test:
        pred = predict(doc, alpha)
        if pred == 0:
            TN += 1
        else:
            FP += 1
    return TP, FP, TN, FN