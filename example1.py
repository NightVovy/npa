from ncpol2sdpa import generate_operators, SdpRelaxation

# Number of operators
n_vars = 2
# Level of relaxation
level = 2

# Get Hermitian operators
X = generate_operators('X', n_vars, hermitian=True)

# Define the objective function
obj = X[0] * X[1] + X[1] * X[0]

# Inequality constraints
inequalities = [-X[1] ** 2 + X[1] + 0.5 >= 0]

# Simple monomial substitutions
substitutions = {X[0]**2: X[0]}

# Obtain SDP relaxation
sdpRelaxation = SdpRelaxation(X)
sdpRelaxation.get_relaxation(level, objective=obj, inequalities=inequalities,
                             substitutions=substitutions)
sdpRelaxation.solve()
print(sdpRelaxation.primal, sdpRelaxation.dual, sdpRelaxation.status)