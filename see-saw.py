import picos as pc
import numpy as np
from numpy import kron, eye
import matplotlib.pyplot as plt

N_ROUNDS = 20
N_STARTS = 70
N_ALPHAS = 10
SOLVER = "cvxopt"


# --------------------------------------------------------------------------------------
# These are just utilitary functions that I use to set the initial random observables.
# The actual implementation of the see-saw procedure is below.
# --------------------------------------------------------------------------------------

def dag(matrix):
    return matrix.conj().T


def outer(vec1, vec2=None):
    """
    Outer product (with complex conjugation) between `vec1` and `vec2`
    If `vec2` is not supplied, return outer product of `vec1` with itself
    """

    if vec1.ndim == 1:
        vec1 = vec1[:, None]
    if vec2:
        if vec2.ndim == 1:
            vec2 = vec2[:, None]
    else:
        vec2 = vec1
    return vec1 @ dag(vec2)


def random_unitary_haar(dim=2):
    """
    Random unitary matrix according to Haar measure.
    Ref.: https://arxiv.org/abs/math-ph/0609050v2
    """
    q, r = np.linalg.qr(randnz((dim, dim)))
    m = np.diagonal(r)
    m = m / np.abs(m)
    return np.multiply(q, m, q)


def random_pure_state(dim=2, density=True):
    """Generates a random pure quantum state of dimension `dim` in Haar measure."""

    st = random_unitary_haar(dim)[:, 0]
    if density:
        st = outer(st)
    return st


def randnz(shape, norm=1 / np.sqrt(2)):
    """Normally distributed complex number matrix (Ginibre ensemble)."""
    real = np.random.normal(0, 1, shape)
    imag = 1j * np.random.normal(0, 1, shape)
    return (real + imag) * norm


# --------------------------------------------------------------------------
# Here starts the implementation of the actual problem.
# --------------------------------------------------------------------------

def random_observables(n=1):
    """Only works for qubit {-1,1} observables: O = Id. - |psi><psi|"""
    return [pc.Constant(2 * random_pure_state() - eye(2)) for _ in range(n)]


def largest_eigenvector(oper):
    eigvals, eigvecs = np.linalg.eig(oper)
    return outer(eigvecs[np.argmax(eigvals)])  # Density matrix format.


def inequality_operator(alpha, rho, A0, A1, B0, B1):
    """Expression must have picos variables, otherwise @ and * will get mixed up!"""
    return rho * (alpha * A0 @ eye(2) + A0 @ B0 + A1 @ B0 + A0 @ B1 - A1 @ B1)


def initial_observable(alpha):
    """Returns the G associated to a tilted CHSH for random observables."""
    A0, A1, B0, B1 = random_observables(4)
    return inequality_operator(alpha, eye(4), A0, A1, B0, B1)


def optimize_observables(alpha, rho, X0, X1, side, verb=0):
    """Optimize the tilted CHSH over either `alice` or `bob` side."""
    prob = pc.Problem()
    X0, X1 = pc.Constant(X0), pc.Constant(X1)
    O = [pc.HermitianVariable(f"O({i})", 2) for i in range(2)]
    prob.add_list_of_constraints([o + eye(2) >> 0 for o in O])
    prob.add_list_of_constraints([eye(2) - o >> 0 for o in O])
    if side == "alice":
        prob.set_objective("max", pc.trace(inequality_operator(alpha, rho, O[0], O[1], X0, X1)).real)
    elif side == "bob":
        prob.set_objective("max", pc.trace(inequality_operator(alpha, rho, X0, X1, O[0], O[1])).real)
    return prob.solve(solver=SOLVER, verbose=verb), pc.Constant(O[0].value), pc.Constant(O[1].value)


def see_saw(alpha, N_ROUNDS=N_ROUNDS, verb=0):
    """See-saw optimization for one random initial setting."""
    A0, A1, B0, B1 = random_observables(4)
    for _ in range(N_ROUNDS):
        rho = largest_eigenvector(inequality_operator(alpha, eye(4), A0, A1, B0, B1))
        prob, B0, B1 = optimize_observables(alpha, rho, A0, A1, side="bob")
        prob, A0, A1 = optimize_observables(alpha, rho, B0, B1, side="alice")
    return prob


def driver(N_ALPHAS=N_ALPHAS, N_STARTS=N_STARTS):
    """Runs the alg. for `N_ALPHAS` values and `N_STARTS` initial settings for each alpha."""
    alphas, optvals = np.linspace(0, 1, N_ALPHAS), []
    for alpha in alphas:
        print(f"\nComputing alpha = {alpha}.")
        best_value = 0
        for _ in range(N_STARTS):
            new_value = see_saw(alpha).value
            if new_value > best_value:  # Take the best out of N_STARTS runs.
                best_value = new_value
                print(f"\r   Best value = {best_value}", end="")
        optvals.append(best_value)
    return alphas, optvals


if __name__ == "__main__":
    driver()
    plt.show()

# --------------------------------------------------------------------------
# Here starts the implementation of the actual problem.
# --------------------------------------------------------------------------