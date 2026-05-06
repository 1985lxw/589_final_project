from sklearn import datasets
import pandas as pd
import numpy as np
import json

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

def load_handwriting_dataset():
    with open('res/table_results.tgn', 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data['rows_views'][0]) 
    digits = df.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    N = len(digits_dataset_X)

    return digits_dataset_X, digits_dataset_y
    
    # Prints the 64 attributes of a random digit, its class,
    # and then shows the digit on the screen
    # digit_to_show = np.random.choice(range(N),1)[0]
    # print("Attributes:", digits_dataset_X[digit_to_show])
    # print("Class:", digits_dataset_y[digit_to_show])
    
    # plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8,8)))
    # plt.show()