import L1_Norm_TTD_AltConvPro
import TTD
import utils
import numpy as np
from copy import deepcopy


# np.random.seed(0)
size = (10, 20, 50)
X = np.random.rand(10, 20, 50) * 40 - 20
num_out = [1000, 2000, 5000]
for num in num_out:
    print("Number of outliers: ", num)
    X_hat = deepcopy(X)
    f_indices = np.random.randint(0, np.prod(size), num)
    for i in f_indices:
        X_hat[np.unravel_index(i, size)] = np.random.randint(40, 60) * np.random.choice([-1, 1])
    

    # L1_Norm_TTD_AltConvPro
    G, losses, t = L1_Norm_TTD_AltConvPro.L1_TTD_AltConvPro(X_hat)
    print(f"L1 With outlier: {np.linalg.norm(X_hat - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))}")
    print(f"L1 Without outlier: {np.linalg.norm(X - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))}")

    # TTD
    G_list, r = TTD.TTD(X_hat)
    X_reconstructed = TTD.TTD_reconstruct(G_list)
    print(f"L2 With outlier: {np.linalg.norm(X_hat - X_reconstructed)}")
    print(f"L2 Without outlier: {np.linalg.norm(X - X_reconstructed)}")

    # Difference
    print(f"Difference L1: {np.linalg.norm(L1_Norm_TTD_AltConvPro.TTD_reconstruct(G) - X_reconstructed)}")