import numpy as np
import cvxpy as cp

from buildHierarchy import build_hierarchy
from value import value


# TODO: where is sdp?

def postCHSH_Q(eta, chshvalue):
    # eta is a 1*4 vectors, represent the successful probability of conclusive
    # event [eta_A0B0,eta_A0B1,eta_A0B0,eta_A1B1]
    # vector[X1,X2,X3]used to represent operators;
    # [0,0,0]identity;
    # [0,1,0]E^0;outcome 0
    # [0,1,1]E^1;outcome 1
    # [1,0,0] A_0^0
    # [1,0,1] A_0^1 ; A_0^2=[0,0,0]-[1,0,0]-[1,0,1]
    # [2,0,0] B_0^0
    # [1,0,0;2,0,0] A_0^0B_0^0
    # [0,1,0;1,0,0] A_0^0E^0
    # chose parameters (custom-made)
    na = 2  # if Alice's outcome number is Na, put inside na=Na-1
    na_in = 4
    # Alice 4 measurements each has 3 outcomes hmm
    nb = 2
    nb_in = 4
    nc_in = 0
    nc = 0
    Q = 2  # Q used to chose level
    Y = build_hierarchy(Q, na_in, na, nb_in, nb, nc_in, nc)  # Y(party,measurement)

    Max = 5  # TODO: what is this?
    T = np.zeros((len(Y), len(Y)))  # 生成一个从 0 到 len(Y)-1 的序列
    for i in range(len(Y)):
        for j in range(i, len(Y)):
            tempY = Y[i][j]
            rowY, colY = tempY.shape
            T[i, j] = 0
            for ir in range(rowY):
                for jc in range(colY):
                    T[i, j] += tempY[ir, jc] * Max ** (ir * colY + jc)
            T[j, i] = T[i, j]

    party = np.unique(T)  # all the different items in T
    B = np.zeros(T.shape, dtype=int)  # reduced the index matrix of T TODO: is int?
    for i in range(len(T)):
        for j in range(len(T)):
            B[i, j] = np.where(party == T[i, j])[0][0]

    # variable declaration
    G = cp.Variable((len(Y), len(Y)), hermitian=True)  # Gram matrix TODO: is this correct?
    V = cp.Variable(len(party))

    # Objective function
    # example: P(0,0|x=0,E)
    # objAE=-real(V(value([0,1,0;1,0,0],party))+V(value([1,0,1],party))-V(value([0,1,0;1,0,1],party)));
    # obj1 = cp.real(V[B[1, 1]])  # obj1=real(V(value([1,1,0;2,1,0],party)));%00
    # obj2 = cp.real(V[B[1, 1]] - V[B[1, 1]])  # TODO: still not used in Matlab
    # obj3 = cp.real(V[B[2, 1]] - V[B[1, 1]])
    # obj4 = cp.real(1 - V[B[1, 1]] - V[B[2, 1]] + V[B[1, 1]])
    # objAE = -cp.real(obj1)
    obj1 = cp.real(V[value([[1, 1, 0], [2, 1, 0]], party)])  # TODO: this may not correct
    obj2 = cp.real(V[value([1, 1, 0], party)] - V[value([[1, 1, 0], [2, 1, 0]], party)])
    obj3 = cp.real(V[value([2, 1, 0], party)] - V[value([[1, 1, 0], [2, 1, 0]], party)])
    obj4 = cp.real(1 - V[value([1, 1, 0], party)] - V[value([2, 1, 0], party)] + V[value([[1, 1, 0], [2, 1, 0]], party)])
    objAE = -cp.real(obj1)

    # Bell-operators
    # CH1 = V[B[1, 0]] + V[B[1, 1]] + V[B[1, 1]] - V[B[1, 1]] - V[B[1, 0]] - V[B[2, 0]]
    # # CH1=V(value([1,0,0;2,0,0],party))+V(value([1,0,0;2,1,0],party))+V(value([1,1,0;2,0,0],party))
    # #   -V(value([1,1,0;2,1,0],party))-V(value([1,0,0],party))-V(value([2,0,0],party));
    # CH2 = V[B[1, 2]] + V[B[1, 3]] + V[B[1, 3]] - V[B[1, 3]] - V[B[1, 2]] - V[B[2, 2]]
    # CH3 = V[B[1, 1]] + V[B[1, 2]] + V[B[1, 3]] - V[B[1, 2]] - V[B[1, 1]] - V[B[2, 3]]
    # CH4 = V[B[1, 3]] + V[B[1, 0]] + V[B[1, 2]] - V[B[1, 0]] - V[B[1, 3]] - V[B[2, 1]]
    # partial = V[B[1, 1]] + V[B[1, 3]] + V[B[2, 1]] + V[B[2, 3]]
    # 计算 CH1
    CH1 = (V[value([[1, 0, 0], [2, 0, 0]], party)] +
           V[value([[1, 0, 0], [2, 1, 0]], party)] +
           V[value([[1, 1, 0], [2, 0, 0]], party)] -
           V[value([[1, 1, 0], [2, 1, 0]], party)] -
           V[value([1, 0, 0], party)] -
           V[value([2, 0, 0], party)])

    # 计算 CH2
    CH2 = (V[value([[1, 2, 0], [2, 2, 0]], party)] +
           V[value([[1, 2, 0], [2, 3, 0]], party)] +
           V[value([[1, 3, 0], [2, 2, 0]], party)] -
           V[value([[1, 3, 0], [2, 3, 0]], party)] -
           V[value([1, 2, 0], party)] -
           V[value([2, 2, 0], party)])

    # 计算 CH3
    CH3 = (V[value([[1, 1, 0], [2, 3, 0]], party)] +
           V[value([[1, 1, 0], [2, 2, 0]], party)] +
           V[value([[1, 0, 0], [2, 3, 0]], party)] -
           V[value([[1, 0, 0], [2, 2, 0]], party)] -
           V[value([1, 1, 0], party)] -
           V[value([2, 3, 0], party)])

    # 计算 CH4
    CH4 = (V[value([[1, 3, 0], [2, 1, 0]], party)] +
           V[value([[1, 3, 0], [2, 0, 0]], party)] +
           V[value([[1, 2, 0], [2, 1, 0]], party)] -
           V[value([[1, 2, 0], [2, 0, 0]], party)] -
           V[value([1, 3, 0], party)] -
           V[value([2, 1, 0], party)])

    # 计算 partial
    partial = (V[value([1, 1, 0], party)] +
               V[value([1, 3, 0], party)] +
               V[value([2, 1, 0], party)] +
               V[value([2, 3, 0], party)])
    CH = cp.real(CH1 + CH2 - CH3 - CH4 - partial)

    C = [CH == chshvalue]

    # constraints for conclusive events
    # TODO: incorrect
    A0B0s = V[B[1, 0]] + V[B[1, 1]] + V[B[1, 0]] + V[B[1, 1]]
    A0B1s = V[B[1, 0]] + V[B[1, 1]] + V[B[1, 0]] + V[B[1, 1]]
    A1B0s = V[B[1, 1]] + V[B[1, 1]] + V[B[1, 0]] + V[B[1, 1]]
    A1B1s = V[B[1, 1]] + V[B[1, 1]] + V[B[1, 0]] + V[B[1, 1]]

    A0s = V[B[1, 0]] + V[B[1, 1]]
    A1s = V[B[1, 1]] + V[B[1, 1]]
    B0s = V[B[2, 0]] + V[B[2, 1]]
    B1s = V[B[2, 1]] + V[B[2, 1]]
    C += [A0B0s == eta[0], A0B1s == eta[1], A1B0s == eta[2], A1B1s == eta[3]]
    C += [A1s == np.sqrt(eta[0]), B0s == np.sqrt(eta[0]), B1s == np.sqrt(eta[0]), A0s == np.sqrt(eta[0])]

    # normalization is adjusted according to your choice of na, nb, nc
    C += [1 - V[B[0, 1]] >= 0]
    for ii in range(2):
        for jj in range(2):
            C += [1 - V[B[ii, jj]] - V[B[ii, jj]] >= 0]

    # constraints for Gram matrix
    for ja in range(2):
        for rr in range(na):
            C += [cp.real(V[B[1, ja]]) >= 0, cp.real(V[B[1, ja]]) <= 1, cp.imag(V[B[1, ja]]) == 0]
            C += [cp.real(V[B[2, ja]]) >= 0, cp.real(V[B[2, ja]]) <= 1, cp.imag(V[B[2, ja]]) == 0]
            for rb in range(nb):
                for jb in range(2):
                    C += [cp.real(V[B[1, ja]]) >= 0, cp.real(V[B[1, ja]]) <= 1, cp.imag(V[B[1, ja]]) == 0]

    for rE in range(nc):
        C += [cp.real(V[B[0, 1]]) >= 0, cp.imag(V[B[0, 1]]) == 0]
        for rr in range(na):
            for ja in range(2):
                for jb in range(2):
                    for rb in range(nb):
                        C += [cp.real(V[B[0, 1]]) >= 0, cp.real(V[B[0, 1]]) <= 1, cp.imag(V[B[0, 1]]) == 0]
                        C += [cp.real(V[B[0, 1]]) >= 0, cp.real(V[B[0, 1]]) <= 1, cp.imag(V[B[0, 1]]) == 0]
                        C += [cp.real(V[B[0, 1]]) >= 0, cp.real(V[B[0, 1]]) >= 0, cp.imag(V[B[0, 1]]) >= 0]

    C += [V[0] == 0, V[1] == 1]
    for i in range(len(party)):
        C += [cp.real(V[i]) >= -1]
        C += [cp.real(V[i]) <= 1]
    for i in range(len(T)):
        for j in range(i, len(T)):
            C += [G[i, j] == V[B[i, j]]]

    C += [G >> 0]

    # Solve the problem
    prob = cp.Problem(cp.Minimize(objAE), C)
    prob.solve(solver="SDPA", verbose=True)

    Gram = G.value
    Pguess = -objAE.value
    return Pguess, Gram


# 示例用法
eta = [1, 1, 1, 1]
chshvalue = 0.004747
Pguess, Gram = postCHSH_Q(eta, chshvalue)
print("Pguess:", Pguess)
print("Gram:", Gram)
