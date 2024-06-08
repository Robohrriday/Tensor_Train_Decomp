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

def L1_TTD_AltConvPro(X:np.array, ranks: list = [None]):
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
    assert d >= 2, "Input tensor must be atleast 2-way tensor"
    G_list = []
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
    for i in range(d-1):
        X = utils.MATLAB_reshape(X, (r[i]*n[i], -1))
        U, V, loss = utils.AltConvPro_LP(X, r[i+1])
        G_list.append(utils.MATLAB_reshape(U, (r[i], n[i], r[i+1])))
        X = V.T
        losses.append(loss[-1])
    G_list.append(utils.MATLAB_reshape(X, (r[d-1], n[d-1], r[d])))
    return G_list, losses

# # Example
# np.random.seed(0)
# X = np.random.rand(5, 10, 15, 20)
# G, losses = L1_TTD_AltConvPro(X)
# # print(X)
# # print(TTD_reconstruct(G))
# print(np.linalg.norm(X - TTD_reconstruct(G))/np.linalg.norm(X))
# print([g.shape for g in G])
# # plt.plot(losses)
# # plt.grid(alpha = 0.5)
# # plt.show()