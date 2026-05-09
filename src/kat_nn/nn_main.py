import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import *
from neural_network import *
from eval_nn import *

def format_for_nn_training(D):
    X = D[0]
    y = D[1]
    return np.hstack([X, y.reshape(-1, 1)])

architectures = [
    (4,),
    (8,),
    (16,),
    (16, 8),
    (32, 16),
    (32, 16, 8),
    (64, 32, 16, 8),
]

alphas = [0.001, 0.01, 0.05, 0.1]
lambdas = [0.001, 0.01, 0.1, 1.0]

datasets = {
    "digits": format_for_nn_training(load_digits_dataset()),
    "parkinsons": format_for_nn_training(load_parkinsons_dataset()),
    "rice": format_for_nn_training(load_rice_dataset()),
    "credit": format_for_nn_training(load_credit_dataset(hot_encoding=True))
}

for dataset in datasets.keys():
    df = datasets[dataset]

    if dataset == "digits":
        cost_fn = categorical_cost
        activation = softmax
    else:
        cost_fn = cost
        activation = sigmoid

    batch_size = min(512, max(64, len(df) // 20))

    print(f"===== Trained Neural Network for {dataset} =====")
    results_df = stratified_k_fold_validation(
        df,
        architectures=architectures,
        lambdas=lambdas,
        k=5,
        alphas=alphas,
        activation=activation,
        cost_fn=cost_fn,
        num_epochs=512,
        batch_size=batch_size,   
    )

    print(results_df)
    results_df.to_latex(f"2tmp/{dataset}.tex", index=False)

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
        activation=activation,
        train_sizes=[10, 20, 40, 80, 120, 160],
        cost_fn=cost_fn,
        num_epochs=512,
        batch_size=batch_size
    )

    plot_learning_curve(sizes, costs, dataset, title="Learning Curve for Best Model")