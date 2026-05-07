from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def hot_encoder_helper(X):
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns

    ct = ColumnTransformer(
        [('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )

    return ct.fit_transform(X)


def load_digits_dataset():
    """Helper function to load digits dataset, excerpt from project specifications."""
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]

    return digits_dataset_X, digits_dataset_y

def load_parkinsons_dataset():
    df = pd.read_csv("res/parkinsons.csv")
    df = df.to_numpy()
    return df[:, :-1], df[:, -1]

def load_rice_dataset():
    df = pd.read_csv("res/rice.csv")
    X = df.iloc[:, :-1]
    return X, np.where(df.iloc[:, -1].values == "Cammeo", 0, 1)

def load_credit_dataset(hot_encoding=False):
    df = pd.read_csv("res/credit_approval.csv")
    X = df.iloc[:, :-1]
    if hot_encoding:
        X = hot_encoder_helper(X)
    return X, df.iloc[:, -1].values
