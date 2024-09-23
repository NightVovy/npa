from ncpol2sdpa import *

level = 1
P = Probability([2, 2], [2, 2])

CHSH = -P([0], [0], 'A') + P([0, 0], [0, 0]) + P([0, 0], [0, 1]) + \
       P([0, 0], [1, 0]) - P([0, 0], [1, 1]) - P([0], [0], 'B')

objective = -CHSH

sdp = SdpRelaxation(P.get_all_operators())
sdp.get_relaxation(level, objective=objective,
                   substitutions=P.substitutions,
                   extramonomials=P.get_extra_monomials('AB'))

sdp.solve(solver="mosek")
print(sdp.primal)


# -0.20710678143685596?
