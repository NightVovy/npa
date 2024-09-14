# https://ncpol2sdpa.readthedocs.io/en/stable/tutorial.html#defining-and-solving-an-optimization-problem-of-noncommuting-variables
# Defining and Solving an Optimization Problem of Noncommuting Variables
from ncpol2sdpa import *

n_vars = 2 # Number of variables
level = 2  # Requested level of relaxation


X = generate_operators('X', n_vars, hermitian=True)  # why here is functional?
obj_nc = X[0] * X[1] + X[1] * X[0]

inequalities_nc = [-X[1] ** 2 + X[1] + 0.5]
substitutions_nc = {X[0]**2 : X[0]}

sdp_nc = SdpRelaxation(X)
sdp_nc.get_relaxation(level, objective=obj_nc, inequalities=inequalities_nc,
                      substitutions=substitutions_nc)
sdp_nc.solve()
print(sdp_nc.primal, sdp_nc.dual, sdp_nc.status)