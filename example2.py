import cvxopt
from ncpol2sdpa import generate_operators, SdpRelaxation, Probability, define_objective_with_I, maximum_violation

level = 1
I = [[0, -1, 0],
     [-1, 1, 1],
     [0, 1, -1]]
# print(maximum_violation(A_configuration, B_configuration, I, level,extra='AB')

P = Probability([2, 2], [2, 2])
objective = define_objective_with_I(I, P)

CHSH = -P([0], [0], 'A') + P([0, 0], [0, 0]) + P([0, 0], [0, 1]) + \
       P([0, 0], [1, 0]) - P([0, 0], [1, 1]) - P([0], [0], 'B')

objective = -CHSH

sdp = SdpRelaxation(P.get_all_operators())
sdp.get_relaxation(level, objective=objective,
                   substitutions=P.substitutions,
                   extramonomials=P.get_extra_monomials('AB'))
sdp.solve(solver="mosek")
print(sdp.primal)
