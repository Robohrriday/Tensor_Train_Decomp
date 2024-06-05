import numpy as np

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
