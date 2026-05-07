import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_digits_dataset
from neural_network import *
from eval_nn import *

X, y = load_digits_dataset()

print(np.hstack([X, y.reshape(-1, 1)]))

df = np.hstack([X, y.reshape(-1, 1)])

architectures = [
    (4,),
    (8,),
    (16,),
    (16, 8),
    (32, 16),
    (32, 16, 8),
    (64, 32, 16, 8),
]
lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]

alphas = [0.01, 0.1] 

dataset = "digits"

print(f"===== Trained Neural Network for {dataset} =====")
results_df = stratified_k_fold_validation(
    df,
    architectures=architectures,
    lambdas=lambdas,
    k=5,
    alphas=alphas,
    num_epochs=512,
    batch_size=128,   
)

print(results_df)
results_df.to_latex(f"589_final_project/figures/{dataset}.tex", index=False)

best = best_configuration(results_df, metric="f1_mean")
best_arch = eval(best["architecture"]) 
best_lambda = best["lambda"]
best_alpha = best["alpha"]

print("Best architecture:", best_arch)
print("Best lambda:", best_lambda)
print("Best alpha:", best_alpha)

sizes, costs = learning_curve_cost(
    df,
    hidden_layers=best_arch,
    lambda_reg=best_lambda,
    alpha=best_alpha,
    train_sizes=[10, 20, 40, 80, 120, 160],
    num_epochs=1000,
    batch_size=128
)

plot_learning_curve(sizes, costs, dataset, title="Learning Curve for Best Model")