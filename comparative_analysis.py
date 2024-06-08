import L1_Norm_TTD_AltConvPro
import TTD
import utils
import numpy as np

np.random.seed(0)
# L1_Norm_TTD_AltConvPro
X = np.random.randint(-20, 20, (5, 6, 7))
G, losses = L1_Norm_TTD_AltConvPro.L1_TTD_AltConvPro(X)
print(np.linalg.norm(X - L1_Norm_TTD_AltConvPro.TTD_reconstruct(G))/np.linalg.norm(X))

# TTD
X = np.random.randint(-20, 20, (5, 6, 7))
G_list, r = TTD.TTD(X)
X_reconstructed = TTD.TTD_reconstruct(G_list)
print(np.linalg.norm(X - X_reconstructed)/np.linalg.norm(X))

for i in range(len(G)):
    print(f"L1: {G[i]}", f"L2: {G_list[i]}", sep = '\n\n')
    print("######")