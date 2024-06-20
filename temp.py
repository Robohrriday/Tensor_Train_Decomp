import highspy as hs
import numpy as np
from scipy.optimize import linprog
from gurobi_optimods.regression import LADRegression
import gurobipy as gp
from gurobipy import GRB

# Define the LP problem
m, n = (10, 10) # Provide the dimensions of matrix A
A = np.random.randint(10, 20, (1000, 1000))
b = np.random.randint(10, 20, 1000)

# Solve the same linear program using scipy

n, k = A.shape
c = np.concatenate([np.ones(shape = n), np.zeros(shape = k)])
A_ub = np.block([
    [-np.eye(n), A],
    [-np.eye(n), -A]
    ])
b_ub = np.concatenate([b, -b])
bounds = [(0, None)]*n + [(None, None)]*k ########## NEEDS ATTENTION  -np.max(np.max(X, axis = 1), axis = 0)*bound_multiplier
result = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds)
if result.status != 0:
    print(result.status, result.message)
print("###################################################### Scipy ######################################################")
print(result.x[-k:])
print(sum(result.x[:n]))
print("###################################################### Scipy ######################################################")

# Solve using LADRegression


model = LADRegression()
model.fit(A, b)
print("###################################################### LADRegression ######################################################")
print(model.coef_)
print(model.intercept_)
print("###################################################### LADRegression ######################################################")


# Solve using gurobi-optimods

# Create a new model
model = gp.Model("lp")

# Create variables
x = model.addVars(n, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "x")
t = model.addVars(m, lb = 0, ub = GRB.INFINITY, name = "t")

# Set objective
model.setObjective(sum(t), GRB.MINIMIZE)

# Add constraints
for i in range(m):
    s = sum(A[i][j]*x[j] for j in range(n))
    model.addConstr(s - b[i] <= t[i])
    model.addConstr(-s + b[i] <= t[i])

# Optimize model
model.optimize()

# Print the solution
print("###################################################### Gurobi ######################################################")
print(f"x= {[x[i].x for i in range(n)]}")
print(sum([t[i].x for i in range(m)]))
print("###################################################### Gurobi ######################################################")