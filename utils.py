import numpy as np
from gurobi_optimods.regression import LADRegression
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tqdm import tqdm

def mode_n_unfolding(X, n):
    """
    Mode-n unfolding of a tensor X
    ## Inputs
    X: numpy.array
        Input tensor
    n: int
        Mode of unfolding
    ---
    ## Outputs
    X_n: numpy.array
        Mode-n unfolding of tensor X
    """
    return np.reshape(np.moveaxis(X, n, 0), (X.shape[n], -1))

# # Example
# np.random.seed(0)
# n = (3, 4, 2)
# X = np.arange(np.prod(n))
# X = X.reshape(n)
# print(X)
# print()
# for i in range(X.shape[-1]):
#     print(X[:, :, i])
# print()
# print(mode_n_unfolding(X, 0))
# print(mode_n_unfolding(X, 1))
# print(mode_n_unfolding(X, 2))


def mode_n_folding(X_n, n, shape):
    """
    Mode-n folding of a tensor X_n
    ## Inputs
    X_n: numpy.array
        Mode-n unfolding of tensor X
    n: int
        Mode of folding
    shape: tuple
        Shape of the tensor X
    ---
    ## Outputs
    X: numpy.array
        Tensor X
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(n)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(X_n, full_shape), 0, n)

# # Example
# np.random.seed(0)
# n = (3, 4, 2)
# X = np.arange(np.prod(n))
# X = X.reshape(n)
# print(X)
# print()
# for i in range(X.shape[-1]):
#     print(X[:, :, i])
# print()
# X_n = mode_n_unfolding(X, 2)
# print(X_n)
# print(mode_n_folding(X_n, 2, n))


def MATLAB_reshape(X, n):
    """
    MATLAB style reshaping of a tensor X
    ## Inputs
    X: numpy.array
        Input tensor
    n: tuple
        New shape of the tensor
    ---
    ## Outputs
    Y: numpy.array
        Reshaped tensor
    """
    return np.reshape(X, n, order='F')

# # Example
# np.random.seed(0)
# n = (2, 2, 3)
# X = np.arange(np.prod(n))
# X = X.reshape(n)
# print(X)
# print()
# for i in range(X.shape[-1]):
#     print(X[:, :, i])
# print()
# print(MATLAB_reshape(X, (6, 2)))


def AltConvPro_LADRegr(M, rank, iterations: int = 10):
    """
    Alternating Convex Programming for matrix decomposition such that M = UV' under L1 Norm.
    ## Inputs
    M: numpy.array
        Input matrix
    rank: int
        Rank of the decomposition
    ---
    ## Outputs
    (U, V): (numpy.array, numpy.array)
        Decomposed matrices
    """

    # Initialization
    U = np.eye(M.shape[0], rank)
    S = np.eye(rank)
    V = np.zeros((M.shape[1], rank))
    losses = []
    for iter in tqdm(range(iterations)):
        print(iter)
        # Update V
        for j in range(M.shape[1]):
            lad = LADRegression()
            lad.fit(U @ S, M[:, j])
            V[j, :] = lad.coef_
        # Update U
        for j in range(M.shape[0]):
            lad = LADRegression()
            lad.fit(V @ S.T, M[j, :])
            U[j, :] = lad.coef_
        # Normalization
        Nv = np.diag(np.diag(V.T @ V))
        Nu = np.diag(np.diag(U.T @ U))
        V = V @ np.linalg.inv(Nv)
        U = U @ np.linalg.inv(Nu)
        S = Nu @ S @ Nv
        losses.append(np.linalg.norm(M.reshape(-1, 1) - (U @ S @ V.T).reshape(-1, 1), ord = 1))
    return U @ S**(0.5), V @ S**(0.5), losses


# # Example
# np.random.seed(0)
# M = np.random.randint(0, 20, (100, 20))
# rank = 10
# U, V, losses = AltConvPro_LADRegr(M, rank = 10, iterations = 30)
# print(U)
# print(V)
# print()
# print(M)
# print()
# print(U @ V.T)
# print(M.shape[0]*M.shape[1], (M.shape[0] + M.shape[1])*rank)
# plt.plot(losses)
# plt.show()

def LP_solve(X, y, bound_multiplier: int = 5):
    """
    Solve the L1 norm optimization problem using linear programming
    ## Inputs
    X: numpy.array
        Input matrix
    y: numpy.array
        Target vector
    ---
    ## Outputs
    result: numpy.array
        Solution vector
    """
    n, k = X.shape
    c = np.concatenate([np.ones(shape = n), np.zeros(shape = k)])
    A_ub = np.block([
        [-np.eye(n), X],
        [-np.eye(n), -X]
        ])
    b_ub = np.concatenate([y, -y])
    bounds = [(0, None)]*n + [(-np.max(np.max(X, axis = 1), axis = 0)*bound_multiplier, np.max(np.max(X, axis = 1), axis = 0)*bound_multiplier)]*k ########## NEEDS ATTENTION
    result = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds)
    if result.status != 0:
        print(result.status, result.message)
    return result.x[-k:]


def AltConvPro_LP(M, rank, iterations: int = 10):
    """
    Alternating Convex Programming for matrix decomposition such that M = UV' under L1 Norm.
    ## Inputs
    M: numpy.array
        Input matrix
    rank: int
        Rank of the decomposition
    ---
    ## Outputs
    (U, V): (numpy.array, numpy.array)
        Decomposed matrices
    """

    # Initialization
    U = np.eye(M.shape[0], rank)
    S = np.eye(rank)
    V = np.zeros((M.shape[1], rank))
    losses = []
    for iter in tqdm(range(iterations)):
        # print(iter)
        # Update V
        for j in range(M.shape[1]):
            V[j, :] = LP_solve(U @ S, M[:, j])
        # Update U
        for j in range(M.shape[0]):
            U[j, :] = LP_solve(V @ S.T, M[j, :])
        # Normalization
        Nv = np.diag(np.diag(V.T @ V))
        Nu = np.diag(np.diag(U.T @ U))
        V = V @ np.linalg.inv(Nv)
        U = U @ np.linalg.inv(Nu)
        S = Nu @ S @ Nv
        losses.append(np.linalg.norm(M.reshape(-1, 1) - (U @ S @ V.T).reshape(-1, 1), ord = 1)/np.linalg.norm(M.reshape(-1, 1), ord = 1))
    return U @ S**(0.5), V @ S**(0.5), losses

# # Example
# # np.random.seed(1)
# M = np.random.randint(0, 20, (100, 30))
# compression_ratios = np.arange(1, 10, 1)/10
# with open("compression_ratios_int(0,20)_100x30.txt", "w") as f:
#     for r in compression_ratios:
#         rank = round((np.prod(M.shape)*r)/(M.shape[0] + M.shape[1]), 0).astype(int)
#         U, V, losses = AltConvPro_LP(M, rank = rank, iterations = 30)
#         f.write(f"{rank}, {losses[-1]}\n")
#         # print(rank)
#         # print(U)
#         # print(V)
#         # print()
#         # print(M)
#         # print()
#         # print(U @ V.T)
#         # print(M.shape[0]*M.shape[1], (M.shape[0] + M.shape[1])*rank)
#     #     plt.plot(losses, label = f"{r}")
#     # plt.legend()
#     # plt.show()
# f.close()


def IRLS(X, y, iterations: int = 10, reg: float = 1e-4):
    """
    Iteratively Reweighted Least Squares for L1 Norm Optimization
    ## Inputs
    X: numpy.array
        Input matrix
    y: numpy.array
        Target vector
    ---
    ## Outputs
    result: numpy.array
        Solution vector
    """
    n, k = X.shape
    W = np.eye(n)
    result = np.random.rand(k)
    for iter in range(iterations):
        # print(iter)
        err = X @ result - y
        W = np.diag(1/np.where(err < reg, reg, np.abs(err)))
        result = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return result

# # Example
# np.random.seed(0)
# X = np.random.randint(0, 10, (10, 5))
# y = np.random.randint(0, 10, (10, ))
# result = IRLS(X, y, iterations = 20)
# print(X @ result)
# print(y)

def AltIRLS(M, rank, iterations: int = 10):
    """
    Alternating IRLS based matrix decomposition such that M = UV' under L1 Norm.
    ## Inputs
    M: numpy.array
        Input matrix
    rank: int
        Rank of the decomposition
    ---
    ## Outputs
    (U, V): (numpy.array, numpy.array)
        Decomposed matrices
    """

    # Initialization
    U = np.eye(M.shape[0], rank)
    S = np.eye(rank)
    V = np.zeros((M.shape[1], rank))
    losses = []
    for iter in tqdm(range(iterations)):
        # print(iter)
        # Update V
        for j in range(M.shape[1]):
            if iter == 0:
                V[j, :] = LP_solve(U @ S, M[:, j])
            else:    
                V[j, :] = IRLS(U @ S, M[:, j], iterations  = 20)
        # Update U
        for j in range(M.shape[0]):
            if iter == 0:
                U[j, :] = LP_solve(V @ S.T, M[j, :])
            else:
                U[j, :] = IRLS(V @ S.T, M[j, :], iterations  = 20)
        # Normalization
        Nv = np.diag(np.diag(V.T @ V))
        Nu = np.diag(np.diag(U.T @ U))
        V = V @ np.linalg.inv(Nv)
        U = U @ np.linalg.inv(Nu)
        S = Nu @ S @ Nv
        losses.append(np.linalg.norm(M.reshape(-1, 1) - (U @ S @ V.T).reshape(-1, 1), ord = 1)/np.linalg.norm(M.reshape(-1, 1), ord = 1))
    return U @ S**(0.5), V @ S**(0.5), losses

# Example
np.random.seed(0)
M = np.random.randint(0, 20, (100, 30))
compression_ratios = np.arange(1, 10, 1)/10
with open("compression_ratios_int(0,20)_100x30.txt", "w") as f:
    for r in compression_ratios:
        rank = round((np.prod(M.shape)*r)/(M.shape[0] + M.shape[1]), 0).astype(int)
        U, V, losses = AltIRLS(M, rank = rank, iterations = 30)
        f.write(f"AltIRLS,{r},{rank},{losses[-1]}\n")
        # print(rank)
        # print(U)
        # print(V)
        # print()
        # print(M)
        # print()
        # print(U @ V.T)
        # print(M.shape[0]*M.shape[1], (M.shape[0] + M.shape[1])*rank)
        plt.plot(losses, label = f"{r}")
    plt.legend()
    plt.show()
f.close()
