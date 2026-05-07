from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import json

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
    return X, df.iloc[:, -1].values

def load_credit_dataset(hot_encoding=False):
    df = pd.read_csv("res/credit_approval.csv")
    X = df.iloc[:, :-1]
    if hot_encoding:
        X = hot_encoder_helper(X)
    return X, df.iloc[:, -1].values

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