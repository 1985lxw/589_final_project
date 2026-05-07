import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

datasets = {
    "digits": load_digits_dataset(),
    "parkinsons": load_parkinsons_dataset(),
    "rice": load_rice_dataset(),
    "credit": load_credit_dataset(hot_encoding=True)
}

architectures = [
    (4,),
    (8,),
    (16,),
    (16, 8),
    (32, 16),
    (32, 16, 8),
    (64, 32, 16, 8),
]

alphas = [0.0001, 0.01, 0.1, 1.0]
learning_rates = [0.001, 0.01, 0.1, 1.0]



for dataset in datasets.keys():
    print(f"===== Accuracy of sklearn NN for {dataset} =====")
    results_list = []
    X, y = datasets[dataset]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    for arch in architectures:
        for a in alphas:
            for lr in learning_rates:
                mlp = MLPClassifier(
                    hidden_layer_sizes=arch,
                    alpha=a,
                    learning_rate_init=lr,
                    max_iter=1000,
                )
                
                mlp.fit(X_train, y_train)
                acc = accuracy_score(y_test, mlp.predict(X_test))
                
                # Append data to the list
                results_list.append({
                    "Architecture": str(arch),
                    "Alpha": a,
                    "Learning Rate": lr,
                    "Accuracy": acc
                })
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="Accuracy", ascending=False)
    print(results_df)