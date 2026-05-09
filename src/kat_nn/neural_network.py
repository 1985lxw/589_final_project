import numpy as np

def fmt(arr):
    return np.array2string(arr, formatter={'float_kind': lambda x: f"{x:.5f}"})

def fmt_matrix(M):
    return "\n".join(
        "\t" + np.array2string(
            np.asarray(row),
            formatter={'float_kind': lambda x: f"{x:8.5f}"}
        ).replace('[', ' ').replace(']', ' ')
        for row in np.asarray(M)
    )

def sigmoid(z):
    """Sigmoid activation for binary classification."""
    return 1 / (1 + np.exp(-z))

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def add_bias(A):
    return np.vstack([np.ones((1, A.shape[1])), A])

def initialize_parameters(layer_sizes):
    """Returns a list of network weights initialized via Xavier Initialization for better convergence."""
    Thetas = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6 / (in_size + out_size))
        Theta = np.random.uniform(-limit, limit, (out_size, in_size + 1))
        Thetas.append(Theta)
    return Thetas

def forwardprop(X, Thetas, activation):
    """Returns the activation result of forward propogation."""
    # 1. a^(l=1) = x^(i)
    A = X.T  

    activations = []
    outs = []

    # 2. Add a bias term to a^(l=1)
    A = add_bias(A)
    activations.append(A)

    # 3. For each layer k=2...L-1
    for l in range(len(Thetas) - 1):
        # 1. z^(l=k) = theta^(l=k-1) * a^(l=k-1)
        Z = Thetas[l] @ activations[-1]
        outs.append(Z)

        # 2. a^(l=k) = g(z^(l=k))
        # use sigmoif for hidden layers
        A = sigmoid(Z)

        # 3. Adds bias term to a^(l=k)
        A = add_bias(A)
        activations.append(A)

    # 4. z^(l=L) = theta^(l=L-1) * a^(l=L-1)
    ZL = Thetas[-1] @ activations[-1]
    outs.append(ZL)

    # 5. a^(l=L) = g(z^(l=L))
    AL = activation(ZL)
    activations.append(AL)

    # Return a^(l=L)
    return activations, outs
        
def cost(Y, AL, Thetas, lambda_reg=0.0):
    """Returns the regularized error/cost of the network."""
    y_flat = np.asarray(Y).ravel()
    n = len(y_flat)

    # Add epsilon to prevent log(0)
    epsilon = 1e-15
    AL = np.clip(AL, epsilon, 1 - epsilon)

    # Compute J
    J = (1.0 / n) * np.sum(-Y * np.log(AL) - (1.0 - Y) * np.log(1.0 - AL))

    # S = computes the square of all weights of the network (except bias weights) and adds them up
    S = 0.0
    for Theta in Thetas:
        S += np.sum(Theta[:, 1:] ** 2)

    return J + ( lambda_reg/(2.0 *n) * S) 

def categorical_cost(Y, AL, Thetas, lambda_reg=0.0):
    """Returns the categorical cross-entropy cost of the network."""
    n = Y.shape[1] 

    epsilon = 1e-15
    AL = np.clip(AL, epsilon, 1 - epsilon)
    
    J = - (1.0 / n) * np.sum(Y * np.log(AL))
    
    S = sum(np.sum(T[:, 1:] ** 2) for T in Thetas)
    return J + (lambda_reg / (2.0 * n) * S)

def backprop(X, Y, Thetas, activation, alpha=1.0, lambda_reg=0.0, verbose=False):
    """Returns the gradients for all layers using full-batch backpropagation."""
    n = X.shape[0]
    L = len(Thetas)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # 1.1 Propagate x^(i) and compute each of the network's outputs, f_theta(x^(i))
    activations, outs = forwardprop(X, Thetas, activation)

    # 1.2 Compute the delta values of all output neurons
    delta = [None] * L
    delta[-1] = activations[-1] - Y

    # 1.3 FOr each network layer, compute the delta values of all neurons in the hidden layers
    for l in range(L - 2, -1, -1):
        A = activations[l+1][1:, :]
        delta[l] = (Thetas[l + 1][:, 1:].T @ delta[l + 1]) * (A * (1 - A))
                

    D = []
    for l in range(L):
        # 1.4 For each network layer, updates gradients of the weights of each layer, based on the current training instance
        grad = (delta[l] @ activations[l].T)

        # 2 For each netowrk layer, compute the final (regularized) gradients of the weights of each layer
        P = np.zeros_like(Thetas[l])
        P[:, 1:] = lambda_reg * Thetas[l][:, 1:]
        grad = (1/n) * (grad + P)

        D.append(grad)

    updated_thetas = []

    for theta, grad in zip(Thetas, D):
        updated_thetas.append(theta - alpha * grad)

    return updated_thetas

def make_mini_batches(X, Y, batch_size):
    """Returns mini batches."""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    X = X[indices]
    if Y.ndim == 1:
        Y = Y[indices]
    else:
        Y = Y[:, indices]

    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = X[start:end]

        if Y.ndim == 1:
            Yb = Y[start:end]
        elif Y.shape[1] == n:
            Yb = Y[:, start:end]
        else:
            Yb = Y[start:end]

        batches.append((Xb, Yb))

    return batches

def nn(X, Y, layer_sizes, activation, cost_fn, alpha=1.0, lambda_reg=0.0, num_epochs=1000, batch_size=None, stop_epsilon=1e-4, verbose=False):
    """Returns the result of training a neural network."""
    Thetas = initialize_parameters(layer_sizes)
    costs = []
    best_cost = np.inf

    for epoch in range(num_epochs):
        if batch_size is None:
            batches = [(X, Y)]
        else:
            batches = make_mini_batches(X, Y, batch_size=batch_size)

        for Xb, Yb in batches:
            Ab, _ = forwardprop(Xb, Thetas, activation)
            if Yb.ndim == 1: 
                Y_cost = Yb.reshape(1, -1) 
            else: 
                Y_cost = Yb            
            current_cost = cost_fn(Y_cost, Ab[-1], Thetas, lambda_reg)
            costs.append(current_cost)
            Thetas = backprop(Xb, Yb, Thetas, activation, alpha, lambda_reg, verbose)

            if best_cost is not None:
                improvement = np.absolute(best_cost - current_cost)
                if improvement < stop_epsilon:
                    if verbose:
                        print(f"Stopping early at epoch {epoch + 1}: improvement {improvement:.6e} < {stop_epsilon}")
                    break

            best_cost = min(best_cost, current_cost)

    return Thetas, current_cost
