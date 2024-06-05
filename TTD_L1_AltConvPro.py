import numpy as np
from scipy.optimize import linprog
import utils
import matplotlib.pyplot as plt

# Naive Implementation


def TTD_reconstruct(G_list:list):
    """
    ## Inputs
    G_list: list
        List of core tensors in TT-format
    ---
    ## Outputs
    X: np.array
        Reconstructed tensor
    """
    d = len(G_list)
    n = [G.shape[1] for G in G_list]
    X = G_list[0]
    for i in range(1, d):
        X = np.tensordot(X, G_list[i], axes=([-1], [0]))
    return np.reshape(X, n)

def L1_TTD_AltConvPro(X:np.array, ranks: list = [None], max_iter:int=100, tol:float=1e-4):
    """
    ## Inputs
    L1 Norm Tensor Train Decomposition using Alternating Convex Programming 
    X: numpy.array
        Input tensor
    ranks: list
        List of ranks of core tensors
    max_iter: int (default = 1000)
        Maximum number of iterations
    tol: float (default = 1e-4)
        Tolerance for convergence
    ---
    ## Outputs
    G: list of numpy.array
        Core tensors
    """

    d = len(X.shape)
    n = X.shape
    G_list = []
    best_G_list = []
    losses = []
    
    # Find ranks
    if ranks[0] is not None:
        r = ranks
    else:
        r = [1]
        for i in range(d):
            temp = utils.MATLAB_reshape(X, (np.prod(n[:i+1]), -1))
            r.append(np.linalg.matrix_rank(temp))
    # print(r)
    # Initialize tensor cores
    for i in range(d):
        G_list.append(np.random.rand(r[i], n[i], r[i+1]))
    
    # Optimization loop
    consecutive_loss = np.inf
    min_loss = consecutive_loss
    iter = 0
    while iter < max_iter and consecutive_loss > tol:
        # Update each core tensor
        # print(iter, G_list)
        for i in range(d):
            if i == 0:
                G_greater_than_i = G_list[1]
                for k in range(2, d):
                    G_greater_than_i = np.tensordot(G_greater_than_i, G_list[k], axes = ([-1], [0]))
                G_greater_than_i = utils.MATLAB_reshape(G_greater_than_i, (-1, r[1]))
                A = G_greater_than_i # @ np.reshape(G_list[0], (r[1], -1))
            elif i == d-1:
                G_less_than_i = G_list[0]
                for k in range(1, d-1):
                    G_less_than_i = np.tensordot(G_less_than_i, G_list[k], axes = ([-1], [0]))
                G_less_than_i = utils.MATLAB_reshape(G_less_than_i, (-1, r[d-1]))
                A = G_less_than_i # @ np.reshape(G_list[-1], (r[d-1], -1))
            else:
                G_less_than_i = G_list[0]
                for k in range(1, i):
                    G_less_than_i = np.tensordot(G_less_than_i, G_list[k], axes = ([-1], [0]))
                G_less_than_i = utils.MATLAB_reshape(G_less_than_i, (-1, r[i]))

                G_greater_than_i = G_list[i+1]
                for k in range(i+2, d):
                    G_greater_than_i = np.tensordot(G_greater_than_i, G_list[k], axes = ([-1], [0]))
                G_greater_than_i = utils.MATLAB_reshape(G_greater_than_i, (-1, r[i+1]))
                A = np.kron(G_greater_than_i, G_less_than_i) # @ np.reshape(G_list[i], (r[i]*r[i+1], -1))
            
            G_i_new = np.random.rand(r[i]*r[i+1], n[i])
            
            for j in range(n[i]):
                c = np.concatenate([np.ones(shape = np.prod(n)//n[i]), np.zeros(shape = r[i]*r[i+1])])
                A_ub = np.block([
                    [-np.eye(np.prod(n)//n[i]), A],
                    [-np.eye(np.prod(n)//n[i]), -A]
                    ])
                b_ub = np.concatenate([utils.mode_n_unfolding(X, i)[j, :], -utils.mode_n_unfolding(X, i)[j, :]])
                bounds = [(0, None)]*(np.prod(n)//n[i]) + [(-1e+1, 1e+1)]*(r[i]*r[i+1])
                result = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds)
                G_i_new[:, j] = result.x[-r[i]*r[i+1]:]
            G_list[i] = utils.mode_n_folding(G_i_new.T, 1, (r[i], n[i], r[i+1]))


        # Compute loss
        loss = np.linalg.norm(np.reshape(X - TTD_reconstruct(G_list), (-1, 1)), ord = 1)/np.linalg.norm(np.reshape(X, (-1, 1)), ord = 1)
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            best_G_list = G_list
        print(iter, loss, sep = ": ")
        if iter == 0:
            consecutive_loss = loss
            prev_loss = loss
        else:
            consecutive_loss = np.abs(loss - prev_loss)
            prev_loss = loss
        iter += 1
    return G_list, losses, best_G_list

# Example
# np.random.seed(3)
# X = np.random.randint(0, 10, (2, 3, 4))
# G, losses, best_G = L1_TTD_AltConvPro(X)
# plt.plot(losses)
# plt.title("Loss vs Iteration\nAvg Loss: {:.4f}".format(np.mean(losses)))
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.grid(alpha = 0.5)
# plt.show()
# print(X)
# print(TTD_reconstruct(best_G))