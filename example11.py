# https://ncpol2sdpa.readthedocs.io/en/stable/tutorial.html#defining-a-polynomial-optimization-problem-of-commuting-variables
from ncpol2sdpa import *

n_vars = 2 # Number of variables
level = 2  # Requested level of relaxation
x = generate_variables('x', n_vars)

obj = x[0]*x[1] + x[1]*x[0]
inequalities = [-x[1]**2 + x[1] + 0.5>=0]

substitutions = {x[0]**2 : x[0]}

sdp = SdpRelaxation(x)
sdp.get_relaxation(level, objective=obj, inequalities=inequalities,
                   substitutions=substitutions)

sdp.solve()
print(sdp.primal, sdp.dual, sdp.status)