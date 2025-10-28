import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence

# Define objective to minimize
def objective(x):
    x1, x2 = x
    return (x1 - 2)**2 + (x2 - 3)**2 + x1 + x2

# Define search space
space = [
    Real(0.0, 5.0, name='x1'),
    Real(0.0, 5.0, name='x2')
]

res = gp_minimize(
    objective,   # function to minimize
    space,       # bounds
    n_calls=50,  # number of evaluations
    random_state=42,
    acq_func = "LCB",
    n_initial_points=1
    
)

print("Best parameters: x1={}, x2={}".format(res.x[0], res.x[1]))
print("Minimum value:", res.fun)

# (Optional) plot how convergence behaves
plot_convergence(res)