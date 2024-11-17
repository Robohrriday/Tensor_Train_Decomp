import implementations.L1_Norm_TTD_AltConvPro as L1_Norm_TTD_AltConvPro
import implementations.L1_Norm_TTD_DivNConq as L1_Norm_TTD_DivNConq
import TTD
import implementations.utils as utils
import numpy as np
from copy import deepcopy

setup = input("Setup: ")
if setup == "1":
        
    ######################## SETUP 1 ########################

    print("********************** SETUP 1 *********************")
    # np.random.seed(0)
    # Size of tensor
    size = (10, 30, 20)
    # Generate random tensor
    X = np.random.randint(10, 20, size)
    # Fraction of outliers as compared to total number of entries in the tensor
    f_num_out = [0, 0.1, 0.2, 0.3]
    num_out = [int(f * np.prod(size)) for f in f_num_out]

    # TT-Ranks
    r = [1]
    for i in range(len(size)):
        temp = utils.MATLAB_reshape(X, (np.prod(size[:i+1]), -1))
        r.append(np.linalg.matrix_rank(temp))
    compression = False

    # Reducing TT-Ranks (new_rank = comp_rat * TT_rank)
    # Comment this block to use original TT-Ranks
    comp_rat = 0.3
    for i in range(1, len(r)-1):
        r[i] = int(comp_rat * r[i])
    compression = True
    assert np.array([r]) >= 1, "All TT-Ranks should be greater than or equal to 1"

    # Print General Information
    print(f"\nTensor Size: {size}")
    print(f"TT-Ranks: {r}")
    print(F"Number of Entries in Original Tensor: {np.prod(size)}")
    print(f"Total Number of Entries in TT-Cores after Compression: {np.sum([r[i] * r[i+1] * size[i] for i in range(len(size))])}" if compression else f"Total Number of Entries in TT-Cores: {np.sum([r[i] * r[i+1] * size[i] for i in range(len(size))])}")
    print(f"Frobenius Norm of Original Tensor: {np.linalg.norm(X)}")

    # Run for different number of outliers
    for iter, num in enumerate(num_out):
        print(f"\n         Number of outliers (frac = {f_num_out[iter]}): {num}\n")

        # X_hat = corrupted tensor with outliers 
        X_hat = deepcopy(X)

        # Add outliers uniformly at random
        if num != 0:
            f_indices = np.random.randint(0, np.prod(size), num)
            for i in f_indices:
                X_hat[np.unravel_index(i, size)] = np.random.randint(100, 200) * np.random.choice([-1, 1])

        print(f"Frobenius Norm of Corrupted Tensor: {np.linalg.norm(X_hat)}")
        # L1_Norm_TTD_AltConvPro
        print("\n############ L1_Norm_TTD_AltConvPro ############\n")
        G, losses, t = L1_Norm_TTD_AltConvPro.L1_TTD_AltConvPro(X_hat, ranks=r)
        print(f"Reconstruction Error (corrupted tensor):    {np.linalg.norm(X_hat - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))}")
        print(f"Reconstruction Error (uncorrupted tensor):  {np.linalg.norm(X - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))}")
        print(f"Max, Min: ", max([np.max(g.ravel()) for g in G]), min([np.min(g.ravel()) for g in G]))
        print(f"Number of close-to-zero values (tol = 1e-4):    {np.sum([np.sum(np.abs(g.ravel()) < 1e-4) for g in G])}")
        print(f"Time: {t}s")

        # TTD
        print("\n############ L2_Norm_TTD_TTSVD ############\n")
        G_list, r1, t = TTD.TTD(X_hat, ranks=r)
        X_reconstructed = TTD.TTD_reconstruct(G_list)
        print(f"Reconstruction Error (corrupted tensor):    {np.linalg.norm(X_hat - X_reconstructed)}")
        print(f"Reconstruction Error (uncorrupted tensor):  {np.linalg.norm(X - X_reconstructed)}")
        print(f"Max, Min: ", max([np.max(g.ravel()) for g in G_list]), min([np.min(g.ravel()) for g in G_list]))
        print(f"Number of close-to-zero values (tol = 1e-4):    {np.sum([np.sum(np.abs(g.ravel()) < 1e-4) for g in G_list])}")
        print(f"Time: {t}s")
elif setup == "2":
    ######################## SETUP 2 ########################

    print("********************** SETUP 2 *********************")
    # np.random.seed(0)
    # Size of tensor
    size = (10, 20, 30) # For 3rd order tensor only
    # Generate random tensor
    tensor_rank = 4
    factor_mat = []
    for i in range(len(size)):
        factor_mat.append(np.random.randint(-5, 5, (size[i], tensor_rank)))
    X = np.zeros(size)
    for i in range(tensor_rank):
        X += np.outer(np.outer(factor_mat[0][:, i], factor_mat[1][:, i]), factor_mat[2][:, i]).reshape(size)

    # Fraction of outliers as compared to total number of entries in the tensor
    f_num_out = [0, 0.1, 0.2, 0.3]
    num_out = [int(f * np.prod(size)) for f in f_num_out]

    # TT-Ranks
    r = [1]
    for i in range(len(size)):
        temp = utils.MATLAB_reshape(X, (np.prod(size[:i+1]), -1))
        r.append(np.linalg.matrix_rank(temp))
    compression = False

    # Reducing TT-Ranks (new_rank = comp_rat * TT_rank)
    # Comment this block to use original TT-Ranks
    comp_rat = 0.3
    for i in range(1, len(r)-1):
        r[i] = int(comp_rat * r[i])
    compression = True

    # Print General Information
    print(f"\nTensor Size: {size}")
    print(f"Tensor Rank: {tensor_rank}")
    print(f"TT-Ranks: {r}")
    print(F"Number of Entries in Original Tensor: {np.prod(size)}")
    print(f"Total Number of Entries in TT-Cores after Compression: {np.sum([r[i] * r[i+1] * size[i] for i in range(len(size))])}" if compression else f"Total Number of Entries in TT-Cores: {np.sum([r[i] * r[i+1] * size[i] for i in range(len(size))])}")
    print(f"Frobenius Norm of Original Tensor: {np.linalg.norm(X)}")

    # Run for different number of outliers
    for iter, num in enumerate(num_out):
        print(f"\n         Number of outliers (frac = {f_num_out[iter]}): {num}\n")

        # X_hat = corrupted tensor with outliers 
        X_hat = deepcopy(X)

        # Add outliers uniformly at random
        if num != 0:
            f_indices = np.random.randint(0, np.prod(size), num)
            for i in f_indices:
                X_hat[np.unravel_index(i, size)] = np.random.randint(500, 600) * np.random.choice([-1, 1])

        print(f"Frobenius Norm of Corrupted Tensor: {np.linalg.norm(X_hat)}")
        # L1_Norm_TTD_AltConvPro
        print("\n############ L1_Norm_TTD_AltConvPro ############\n")
        G, losses, t = L1_Norm_TTD_AltConvPro.L1_TTD_AltConvPro(X_hat, ranks=r)
        print(f"Reconstruction Error (corrupted tensor):    {np.linalg.norm(X_hat - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))}")
        print(f"Reconstruction Error (uncorrupted tensor):  {np.linalg.norm(X - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))}")
        print(f"Max, Min: ", max([np.max(g.ravel()) for g in G]), min([np.min(g.ravel()) for g in G]))
        print(f"Number of close-to-zero values (tol = 1e-4):    {np.sum([np.sum(np.abs(g.ravel()) < 1e-4) for g in G])}")
        print(f"Time: {t}s")

        # TTD
        print("\n############ L2_Norm_TTD_TTSVD ############\n")
        G_list, r1, t = TTD.TTD(X_hat, ranks=r)
        X_reconstructed = TTD.TTD_reconstruct(G_list)
        print(f"Reconstruction Error (corrupted tensor):    {np.linalg.norm(X_hat - X_reconstructed)}")
        print(f"Reconstruction Error (uncorrupted tensor):  {np.linalg.norm(X - X_reconstructed)}")
        print(f"Max, Min: ", max([np.max(g.ravel()) for g in G_list]), min([np.min(g.ravel()) for g in G_list]))
        print(f"Number of close-to-zero values (tol = 1e-4):    {np.sum([np.sum(np.abs(g.ravel()) < 1e-4) for g in G_list])}")
        print(f"Time: {t}s")