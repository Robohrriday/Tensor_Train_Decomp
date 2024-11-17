import numpy as np
from gurobi_optimods.regression import LADRegression
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tqdm import tqdm
from copy import deepcopy

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

# Example
if __name__ == "__main__":
    np.random.seed(0)
    n = (3, 4, 2)
    X = np.arange(np.prod(n))
    X = X.reshape(n)
    print(X)
    print()
    for i in range(X.shape[-1]):
        print(X[:, :, i])
    print()
    X_n = mode_n_unfolding(X, 2)
    print(X_n)
    print(mode_n_folding(X_n, 2, n))


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

# Example
if __name__ == "__main__":
    np.random.seed(0)
    n = (2, 2, 3)
    X = np.arange(np.prod(n))
    X = X.reshape(n)
    print(X)
    print()
    for i in range(X.shape[-1]):
        print(X[:, :, i])
    print()
    print(MATLAB_reshape(X, (6, 2)))

def LP_solve(X, y, w):
    """
    Solve the L1 norm optimization problem using linear programming
    ## Inputs
    X: numpy.array
        Input matrix
    y: numpy.array
        Target vector
    w: numpy.array
        Weight vector
    ---
    ## Outputs
    result: numpy.array
        Solution vector
    """
    n, k = X.shape
    c = np.concatenate([w, w, np.zeros(shape = k)])
    A_eq = np.block([np.eye(n), -np.eye(n), X])
    b_eq = y
    bounds = [(0, None)]*n + [(0, None)]*n + [(None, None)]*k ########## NEEDS ATTENTION  -np.max(np.max(X, axis = 1), axis = 0)*bound_multiplier
    result = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = bounds)
    if result.status != 0:
        # print(result.status, result.message)
        return None
    else:
        return result.x[-k:]


def AltConvPro_LP(M, rank, iterations: int = 30, tol: float = 1e-5, sigma:float = 6):
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
    W = np.ones((M.shape[0], M.shape[1]))
    losses = []
    for iter in tqdm(range(iterations)):
        # print(iter)
        # Update V
        for j in range(M.shape[1]):
            temp = LP_solve(U @ S, M[:, j], W[:, j])
            if temp is not None:
                V[j, :] = temp
        # Update U
        for j in range(M.shape[0]):
            temp = LP_solve(V @ S.T, M[j, :], W[j, :])
            if temp is not None:
                U[j, :] = temp
        # Normalization
        Nv = np.diag(V.T @ V)
        Nu = np.diag(U.T @ U)
        V = V @ np.diag(1/Nv)
        U = U @ np.diag(1/Nu)
        S = np.diag(Nu) @ S @ np.diag(Nv)
        W = np.exp(-(M - U @ S @ V.T)**2/(2*sigma**2))
        W = W/np.sum(np.sum(W, axis = 1), axis = 0)
        losses.append(np.linalg.norm((M - (U @ S @ V.T)).reshape(-1, 1), ord = 1)/np.linalg.norm(M.reshape(-1, 1), ord = 1))
        if iter == 0:
            consecutive_loss = losses[-1]
        else:
            consecutive_loss = abs(losses[-1] - losses[-2])
            if consecutive_loss < tol:
                break
        # if iter == 0:
        #     U_old = U
        #     V_old = V
        # else:
        #     if iter == 1:
        #         simU = np.diag(U.T @ U_old)/(np.linalg.norm(U, axis = 0)*np.linalg.norm(U_old, axis = 0))
        #         simV = np.diag(V.T @ V_old)/(np.linalg.norm(V, axis = 0)*np.linalg.norm(V_old, axis = 0))
        #     if all(simU > np.ones(len(simU)) - tol) and all(simV > np.ones(len(simV)) - tol):
        #         break
    return U @ S**(0.5), V @ S**(0.5), losses, W


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


# # np.random.seed(1)
# M = np.random.randint(-20, 20, (100, 30))
# comp_rat = [0.1, 0.3, 0.5, 0.7, 0.9]
# size = (100, 30)

# list_l1_err_w = []
# list_l1_err_wo = []
# list_l2_err_w = []
# list_l2_err_wo = []

# for r in comp_rat:
#     rank = round((np.prod(M.shape)*r)/(M.shape[0] + M.shape[1]), 0).astype(int)
#     A, S, B = np.linalg.svd(M, full_matrices = True)
#     M = A[:, :rank] @ np.diag(S[:rank]) @ B[:rank, :]
#     # U, V, losses = AltConvPro_LP(M, rank = rank, iterations = 30)
#     # X1 = U @ V.T
#     # A, S, B = np.linalg.svd(M, full_matrices = True)
#     # X2 = A[:, :rank] @ np.diag(S[:rank]) @ B[:rank, :]
#     # print(rank)
#     # print("L1 Error: ", np.linalg.norm(M - X1))
#     # print("L2 Error: ", np.linalg.norm(M - X2))
#     # print("Difference: ", np.linalg.norm(X1 - X2))

#     l1_err_w = []
#     l1_err_wo = []
#     l2_err_w = []
#     l2_err_wo = []
#     num_out = [300, 600, 900, 1200, 1500]
#     for num in num_out:
#         print("Number of outliers: ", num)
#         M_hat = deepcopy(M)
#         f_indices = np.random.randint(0, np.prod(size), num)
#         for i in f_indices:
#             M_hat[np.unravel_index(i, size)] = np.random.randint(100, 200) * np.random.choice([-1, 1])
        

#         # L1_Norm_TTD_AltConvPro
#         U, V, losses = AltConvPro_LP(M_hat, rank = rank, iterations = 30)
#         X1 = U @ V.T
#         l1_err_w.append(np.linalg.norm(M_hat - X1))
#         l1_err_wo.append(np.linalg.norm(M - X1))

#         # TTD
#         X, Y, Z = np.linalg.svd(M_hat, full_matrices = False)
#         X = X[:, :rank]
#         Z = Z[:rank, :]
#         X2 = X @ np.diag(Y[:rank]) @ Z
#         l2_err_w.append(np.linalg.norm(M_hat - X2))
#         l2_err_wo.append(np.linalg.norm(M - X2))

#     plt.figure()
#     plt.title("Compression Ratio (fixed): " + str(r) + " Rank (fixed): " + str(rank))
#     plt.xlabel("Number of Outliers")
#     plt.ylabel("Squared Error")
#     plt.plot(num_out, l1_err_w, label = "L1 wrt Corrupted", marker = 'o', linestyle = '-')
#     plt.plot(num_out, l2_err_w, label = "L2 wrt Corrupted", marker = 'x', linestyle = '-')
#     plt.plot(num_out, [np.linalg.norm(M_hat)]*len(num_out), label = "L2 Norm of M_hat", linestyle = '--')
#     plt.plot(num_out, [np.linalg.norm(M)]*len(num_out), label = "L2 Norm of M", linestyle = '--')
#     plt.grid(alpha = 0.5)
#     plt.legend()
#     plt.savefig(".\\plots\\outlier_analysis\\Comp_ratio(fixed)_" + str(r) + "rank(fixed)_" + str(rank) + "w.png")

#     plt.figure()
#     plt.title("Compression Ratio (fixed): " + str(r) + " Rank (fixed): " + str(rank))
#     plt.xlabel("Number of Outliers")
#     plt.ylabel("Squared Error")
#     plt.plot(num_out, l1_err_wo, label = "L1 wrt Uncorrupted", marker = 'o', linestyle = '-')
#     plt.plot(num_out, l2_err_wo, label = "L2 wrt Uncorrupted", marker = 'x', linestyle = '-')
#     plt.plot(num_out, [np.linalg.norm(M_hat)]*len(num_out), label = "L2 Norm of M_hat", linestyle = '--')
#     plt.plot(num_out, [np.linalg.norm(M)]*len(num_out), label = "L2 Norm of M", linestyle = '--')
#     plt.grid(alpha = 0.5)
#     plt.legend()
#     plt.savefig(".\\plots\\outlier_analysis\\Comp_ratio(fixed)_" + str(r) + "rank(fixed)_" + str(rank) + "wo.png")

#     list_l1_err_w.append(l1_err_w)
#     list_l1_err_wo.append(l1_err_wo)
#     list_l2_err_w.append(l2_err_w)
#     list_l2_err_wo.append(l2_err_wo)

# for idx, num in enumerate(num_out):
#     plt.figure()
#     plt.title("Num_out(fixed)_" + str(num))
#     plt.xlabel("Compression Ratio")
#     plt.ylabel("Squared Error")
#     plt.plot(comp_rat, [list_l1_err_w[j][idx] for j in range(len(list_l1_err_w))], label = "L1 wrt Corrupted", marker = 'o', linestyle = '-')
#     plt.plot(comp_rat,  [list_l2_err_w[j][idx] for j in range(len(list_l2_err_w))], label = "L2 wrt Corrupted", marker = 'x', linestyle = '-')
#     plt.plot(comp_rat, [np.linalg.norm(M_hat)]*len(comp_rat), label = "L2 Norm of M_hat", linestyle = '--')
#     plt.plot(comp_rat, [np.linalg.norm(M)]*len(comp_rat), label = "L2 Norm of M", linestyle = '--')
#     plt.grid(alpha = 0.5)
#     plt.legend()
#     plt.savefig(".\\plots\\outlier_analysis\\Num_out(fixed)_" + str(num) + "w.png")

#     plt.figure()
#     plt.title("Num_out(fixed)_" + str(num))
#     plt.xlabel("Compression Ratio")
#     plt.ylabel("Squared Error")
#     plt.plot(comp_rat,  [list_l1_err_wo[j][idx] for j in range(len(list_l1_err_wo))], label = "L1 wrt Uncorrupted", marker = 'o', linestyle = '-')
#     plt.plot(comp_rat,  [list_l2_err_wo[j][idx] for j in range(len(list_l2_err_wo))], label = "L2 wrt Uncorrupted", marker = 'x', linestyle = '-')
#     plt.plot(comp_rat, [np.linalg.norm(M_hat)]*len(comp_rat), label = "L2 Norm of M_hat", linestyle = '--')
#     plt.plot(comp_rat, [np.linalg.norm(M)]*len(comp_rat), label = "L2 Norm of M", linestyle = '--')
#     plt.grid(alpha = 0.5)
#     plt.legend()
#     plt.savefig(".\\plots\\outlier_analysis\\Num_out(fixed)_" + str(num) + "wo.png")




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

# # Example
# np.random.seed(0)
# M = np.random.randint(0, 20, (100, 30))
# compression_ratios = np.arange(1, 10, 1)/10
# with open("compression_ratios_int(0,20)_100x30.txt", "w") as f:
#     for r in compression_ratios:
#         rank = round((np.prod(M.shape)*r)/(M.shape[0] + M.shape[1]), 0).astype(int)
#         U, V, losses = AltIRLS(M, rank = rank, iterations = 30)
#         f.write(f"AltIRLS,{r},{rank},{losses[-1]}\n")
#         # print(rank)
#         # print(U)
#         # print(V)
#         # print()
#         # print(M)
#         # print()
#         # print(U @ V.T)
#         # print(M.shape[0]*M.shape[1], (M.shape[0] + M.shape[1])*rank)
#         plt.plot(losses, label = f"{r}")
#     plt.legend()
#     plt.show()
# f.close()

def P(a: np.array, b: np.array):
    L = np.argsort(a/b)
    found = False
    for i in range(1, L.shape[0]):
        if np.sum(b[L[0:i]]) >= np.sum(b[L[i:]]):
            found = True
            break
    if found:
        return a[L[i-1]]/b[L[i-1]]
    else:
        return a[L[-1]]/b[L[-1]]


def Q(X, y, updating: str):
    I = np.nonzero(y)[0]
    if updating == 'v':
        z = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            z[j] = P(np.multiply(np.sign(y[I]), X[I, j]), np.abs(y[I]))
    else:
        z = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            z[i] = P(np.multiply(np.sign(y[I]), X[i, I]), np.abs(y[I]))
    return z

def DivNConq(X, rank:int, iterations:int = 30, tol:float = 1e-5):
    d, n = X.shape
    U, V, losses = AltConvPro_LP(X, rank = rank, iterations = 1)
    # U = np.ones((d, rank))
    # V = np.ones((n, rank))
    losses = []
    for iter in tqdm(range(iterations)):
        # print(iter)
        for i in range(rank):
            E_i = X - U @ V.T + np.outer(U[:, i], V[:, i]) 
            V[:, i] = Q(E_i, U[:, i], updating = 'v')
            U[:, i] = Q(E_i, V[:, i], updating = 'u')
        losses.append(np.linalg.norm(X.reshape(-1, 1) - (U @ V.T).reshape(-1, 1), ord = 1)/np.linalg.norm(X.reshape(-1, 1), ord = 1))
        if iter == 0:
            consecutive_loss = losses[-1]
        else:
            if iter == 1:
                consecutive_loss = abs(losses[-1] - losses[-2])
            if consecutive_loss < tol:
                break
            consecutive_loss = abs(losses[-1] - losses[-2])
    return U, V, losses
