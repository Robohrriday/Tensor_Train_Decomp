### Implementation of Tensor Train Decomposition using TT-SVD algo from the paper "Tensor-Train Decomposition" by Ivan V. Oseledets

import numpy as np
import scipy.linalg as la
import utils

def TTD(X:np.array, eps:float = 10e-3):
    """
    ## Inputs
    X: np.array
        A d-way tensor `X`
    eps: float
        Error tolerance
    ---
    ## Outputs
    G_list: list
        List of core tensors in TT-format
    r: list
        List of ranks of the core tensors
    """


    d = len(X.shape)
    n = X.shape
    assert d >= 2, "Input tensor must be atleast 2-way tensor"
    G_list = []
    delta = eps/np.sqrt(d-1)
    r = [1] + [0]*(d-1) + [1] 
    for i in range(d-1):
        X = utils.MATLAB_reshape(X, (r[i]*n[i], -1))
        U, S, V = la.svd(X, full_matrices=False)

        norm = np.linalg.norm(S)
        if norm == 0:
            r[i+1] = 1
        elif delta <= 0:
            r[i+1] = len(S)
        else:
            r[i+1] = len(S[np.cumsum(S**2)[::-1] > delta**2])
        U = U[:, :r[i+1]]
        S = S[:r[i+1]]
        V = V[:r[i+1], :]
        G_list.append(utils.MATLAB_reshape(U, (r[i], n[i], r[i+1])))
        X = np.diag(S) @ V
    G_list.append(utils.MATLAB_reshape(X, (r[d-1], n[d-1], r[d])))
    return G_list, r

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

# ### Example
# np.random.seed(0)
# X = np.random.randn(5, 10, 15, 20)
# G_list, r = TTD(X)
# X_reconstructed = TTD_reconstruct(G_list)
# print(np.linalg.norm(X - X_reconstructed)/np.linalg.norm(X))
# print(r)
# print([G.shape for G in G_list])