import numpy as np
import matplotlib.pyplot as plt
import L1_Norm_TTD_AltConvPro
import utils
from time import time

"""
Metrics to analyze:
1) Time Complexity:         Use time library
2) Convergence:             Plot loss vs iterations for multiple runs
3) Sparsity:                Number of close-to-zero-values in the core tensors
4) Robustness to Outliers:  Vary number and magnitude of outliers
5) Compression Ratio:       Vary TT-Ranks of core tensors

Tunable Paramters:
1) Number of Dimensions:    3, 4, 5
2) Size of each Dimension:  5, 10, 20, 50, 100
3) Data Distribution:       uniform_disc, uniform_cont, normal, laplacian, etc.
4) TT-Ranks of Cores:       Compression-Ratio based (0.5, 0.3, 0.1), 2
"""

def form_tensor(size, dist, cr, frac_outliers):
    if dist == "uniform_disc":
        X = np.random.randint(-np.max(size), np.max(size), size)
        if frac_outliers != 0:
            num_out = int(frac_outliers*np.prod(size))
            
    elif dist == "uniform_cont":
        X = np.random.rand(size)
    elif dist == "normal":
        X = np.random.randn(size)
    elif dist == "laplacian":
        X = np.random.laplace(size)
    
    return X


num_dim = [3, 4, 5]
dimension_sizes = {3: [[5, 10, 20], [20, 10, 50], [50, 100, 20], [20, 20, 20]], 4: [[5, 10, 20, 50], [20, 10, 50, 100], [20, 20, 20, 20]], 5: [20, 20, 20, 20, 20]}
data_distributions = ["uniform_disc", "uniform_cont", "normal", "laplacian"]
compression_ratios = ["standard", 0.9, 0.7, 0.5, 0.3, 0.1, 2] # Standard -> Inherent TT-rank, 2 -> Double the Standard TT-rank
frac_outliers = [0.0, 0.1, 0.3, 0.5] # Fraction of total elements
runs = 20

for dim in num_dim:
    for size in dimension_sizes[dim]:
        for dist in data_distributions:
            for cr in compression_ratios:

                analysis_name = f"{dim}D_{size}_{dist}_cr-{cr}_num_out-{frac_outliers}"
                print(f"Performing analysis: {analysis_name}")

                for run in range(runs):
                    X, ranks = form_tensor(size, dist, cr, frac_outliers)
                    G, losses, t = L1_Norm_TTD_AltConvPro.L1_TTD_AltConvPro(X, ranks = ranks)