from sklearn import datasets
import pandas as pd

def load_digits_dataset():
    """Helper function to load digits dataset, excerpt from project specifications."""
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]

    return digits_dataset_X, digits_dataset_y

def load_parkinsons_dataset():
    # TODO: edit as needed
    df = pd.read_csv("res/parkinsons.csv")
    return df.iloc[:, :-1], df.iloc[:, -1]

def load_rice_dataset():
    # TODO: edit as needed
    df = pd.read_csv("res/rice.csv")
    return df.iloc[:, :-1], df.iloc[:, -1]

def load_credit_dataset():
    # TODO: edit as needed
    df = pd.read_csv("res/credit_approval.csv")
    return df.iloc[:, :-1], df.iloc[:, -1]