import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_digits_dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(15,), max_iter=500)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
