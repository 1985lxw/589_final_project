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
    return 1 / (1 + np.exp(-z))

def add_bias(A):
    return np.vstack([np.ones((1, A.shape[1])), A])

def initialize_parameters(layer_sizes):
    """Returns a list of network weights initialized from a Gaussian distribution with mu=0, sigma^2=1."""
    Thetas = []

    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        # Gaussian distribution initialization
        Theta = np.random.randn(out_size, in_size + 1)
        Thetas.append(Theta)

    return Thetas

def forwardprop(X, Thetas):
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
        A = sigmoid(Z)

        # 3. Adds bias term to a^(l=k)
        A = add_bias(A)
        activations.append(A)

    # 4. z^(l=L) = theta^(l=L-1) * a^(l=L-1)
    ZL = Thetas[-1] @ activations[-1]
    outs.append(ZL)

    # 5. a^(l=L) = g(z^(l=L))
    AL = sigmoid(ZL)
    activations.append(AL)

    # Return a^(l=L)
    return activations, outs
        
def cost(Y, AL, Thetas, lambda_reg=0.0):
    """Returns the regularized error/cost of the network."""
    y_flat = np.asarray(Y).ravel()
    n = len(y_flat)

    # Compute J
    J = (1.0 / n) * np.sum(-Y * np.log(AL) - (1.0 - Y) * np.log(1.0 - AL))

    # S = computes the square of all weights of the network (except bias weights) and adds them up
    S = 0.0
    for Theta in Thetas:
        S += np.sum(Theta[:, 1:] ** 2)

    return J + ( lambda_reg/(2.0 *n) * S) 

def backprop(X, Y, Thetas, alpha=1.0, lambda_reg=0.0, verbose=False):
    """Returns the gradients for all layers using full-batch backpropagation."""
    n = X.shape[0]
    L = len(Thetas)
    grads_sum = [np.zeros_like(theta) for theta in Thetas]
    
    if verbose:
        for i in range(n):
            x = X[i:i+1]
            y = Y[i:i+1].T

            # 1.1 Propagate x^(i) and compute each of the network's outputs, f_theta(x^(i))
            activations, outs = forwardprop(x, Thetas)

            # 1.2 Compute the delta values of all output neurons
            delta = [None] * L
            delta[-1] = activations[-1] - y

            print(f"\tComputing gradients based on training instance {i+1}")
            print(f"\t\tdelta{L+1}: {fmt(delta[-1].ravel())}")

            # 1.3 FOr each network layer, compute the delta values of all neurons in the hidden layers
            for l in range(L - 2, -1, -1):
                A = activations[l + 1][1:, :]
                delta[l] = (Thetas[l + 1][:, 1:].T @ delta[l + 1]) * (A * (1 - A))
                print(f"\t\tdelta{l+2}: {fmt(delta[l].ravel())}")

            inst_grads = []
            for l in range(L):
                # 1.4 For each network layer, updates gradients of the weights of each layer, based on the current training instance
                inst_grad = delta[l] @ activations[l].T
                inst_grads.append(inst_grad)
                grads_sum[l] += inst_grad

            for l in range(L - 1, -1, -1):
                print(f"\n\t\tGradients of Theta{l+1} based on training instance {i+1}:")
                print("\t\t" + fmt_matrix(inst_grads[l]).replace("\n", "\n\t\t"))

        D = []
        print("\n\tThe entire training set has been processed. Computing the average (regularized) gradients:")

        for l in range(L):
            # 2 For each netowrk layer, compute the final (regularized) gradients of the weights of each layer
            P = np.zeros_like(Thetas[l])
            P[:, 1:] = lambda_reg * Thetas[l][:, 1:]
            grad = (grads_sum[l] + P) / n
            D.append(grad)

            print(f"\t\tFinal regularized gradients of Theta{l+1}:")
            print("\t\t\t" + fmt_matrix(grad).replace("\n", "\n\t\t\t"))

        return [theta - alpha * grad for theta, grad in zip(Thetas, D)]
    else:
        n = X.shape[0]
        L = len(Thetas)
        Y = Y.reshape(1, -1)

        # 1.1 Propagate x^(i) and compute each of the network's outputs, f_theta(x^(i))
        activations, outs = forwardprop(X, Thetas)

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
    Y = Y[indices]

    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((X[start:end], Y[start:end]))
    return batches

def nn(X, Y, layer_sizes, alpha=1.0, lambda_reg=0.0, num_epochs=1000, batch_size=None, verbose=False):
    """Returns the result of training a neural network."""
    Thetas = initialize_parameters(layer_sizes)
    costs = []

    if verbose:
        print(f"Regularization parameter lambda={lambda_reg:.3f}")
        print(f"\nInitializing the network with the following structure (number of neurons per layer): {layer_sizes}")
       
        # hard coding for testing ex1
        if layer_sizes == [1, 2, 1]:
            Thetas = [
                np.array([[0.4, 0.1],
                        [0.3, 0.2]]),
                np.array([[0.7, 0.5, 0.6]])
            ]
        
        # hard coding for testing ex2
        if layer_sizes == [2, 4, 3, 2]:
            theta1 = np.array([
                [0.42, 0.15, 0.40],
                [0.72, 0.10, 0.54],
                [0.01, 0.19, 0.42],
                [0.30, 0.35, 0.68]
            ])
            
            theta2 = np.array([
                [0.21, 0.67, 0.14, 0.96, 0.87],
                [0.87, 0.42, 0.20, 0.32, 0.89],
                [0.03, 0.56, 0.80, 0.69, 0.09]
            ])
            
            theta3 = np.array([
                [0.04, 0.87, 0.42, 0.53],
                [0.17, 0.10, 0.95, 0.69]
            ])

            Thetas = [theta1, theta2, theta3]

        for i, T in enumerate(Thetas):
            print(f"\nInitial Theta{i+1} (the weights of each neuron, including the bias weight, are stored in the rows):")
            print(fmt_matrix(T))

        print("\nTraining set")
        for i in range(X.shape[0]):
            print(f"\tTraining instance {i+1}")
            print(f"\t\tx: {fmt(X[i].ravel())}")
            print(f"\t\ty: {fmt(np.asarray(Y[i]).ravel())}")

    for epoch in range(num_epochs):
        if batch_size is None:
            batches = [(X, Y)]
        else:
            batches = make_mini_batches(X, Y, batch_size=batch_size)

        epoch_cost = 0.0
        total_seen = 0

        for Xb, Yb in batches:
            if verbose:
                print("\n--------------------------------------------")
                print("Computing the error/cost, J, of the network")

            for i in range(Xb.shape[0]):
                if verbose:
                    x = Xb[i:i+1]
                    y = Yb[i:i+1].T
                    print(f"\tProcessing training instance {i+1}")
                    print(f"\tForward propagating the input {fmt(x.ravel())}")

                    activations, outs = forwardprop(x, Thetas)

                    print(f"\t\ta1: {fmt(activations[0].ravel())}")
                    for l in range(len(outs)):
                        print(f"\n\t\tz{l+2}: {fmt(outs[l].ravel())}")
                        print(f"\t\ta{l+2}: {fmt(activations[l+1].ravel())}")

                    AL = activations[-1]
                    print(f"\n\t\tf(x): {fmt(AL.ravel())}")
                    print(f"\tPredicted output for instance {i+1}: {fmt(AL.ravel())}")
                    print(f"\tExpected output for instance {i+1}: {fmt(np.array([y]))}")
                    J_inst = (-y * np.log(AL) - (1 - y) * np.log(1 - AL))
                    print(f"\tCost, J, associated with instance {i+1}: {np.sum(J_inst):.3f}\n")

            Ab, _ = forwardprop(Xb, Thetas)

            if verbose:
                full_J = cost(Yb, Ab[-1], Thetas, lambda_reg)
                print(f"Final (regularized) cost, J, based on the complete training set: {full_J:.5f}")

            Thetas = backprop(Xb, Yb, Thetas, alpha, lambda_reg, verbose)

            batch_cost = cost(Yb, Ab[-1], Thetas, lambda_reg)
            epoch_cost += batch_cost * Xb.shape[0]
            total_seen += Xb.shape[0]

        epoch_cost /= total_seen
        costs.append(epoch_cost)

    return Thetas, costs
