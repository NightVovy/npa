import numpy as np
import math
from simplifyProjectors import simplify_projectors


def build_hierarchy(Q, na_in, na, nb_in, nb, nc_in, nc):
    """
    构建基于 NPA 层次结构的矩阵 M。

    参数:
    Q (int): 层次结构的级别。
    na_in (int): Alice 的测量次数。
    na (int): Alice 的测量结果数量。
    nb_in (int): Bob 的测量次数。
    nb (int): Bob 的测量结果数量。

    返回:
    list: 包含简化投影矩阵的列表 M。
    """
    # Initialize operators
    op = [[0, 0, 0]]  # identity
    for ia in range(na_in):
        for ja in range(na):
            op.append([1, ia, ja])  # A operator

    for ib in range(nb_in):
        for jb in range(nb):
            op.append([2, ib, jb])  # B operator

    if Q == 2:
        # AB operators
        for ja in range(na_in):
            for ra in range(na):
                for jb in range(na_in):
                    for rb in range(nb):
                        op.append([[1, ja, ra], [2, jb, rb]])  # A0B0

    if Q == 3:
        # AB operators
        for ja in range(na_in):
            for ra in range(na):
                for jb in range(na_in):
                    for rb in range(nb):
                        op.append([[1, ja, ra], [2, jb, rb]])  # A0B0
        # AE moments
        for jc in range(1, nc_in + 1):  # +1?
            for j in range(na_in):
                for k in range(na):
                    for m in range(nc):
                        op.append([[1, j, k], [0, jc, m]])
        # BE moments
        for jc in range(1, nc_in + 1):
            for j in range(nb_in):
                for k in range(nb):
                    for m in range(nc):
                        op.append([[2, j, k], [0, jc, m]])

        # A1A2 operators
        for ra in range(na):
            for rA in range(na):
                op.append([[1, 0, ra], [1, 1, rA]])  # np.array?
        # B1B2 operators
        for rb in range(nb):
            for rB in range(nb):
                op.append([[2, 0, rb], [2, 1, rB]])
        # E1E2 operators
        for jc in range(1, nc_in):
            for jC in range(jc + 1, nc_in + 1):
                for rc in range(nc):
                    for rC in range(nc):
                        op.append([[0, jc, rc], [0, jC, rC]])
        # ABE operators
        for ja in range(na_in):
            for ra in range(na):
                for jb in range(na_in):
                    for rb in range(nb):
                        for jc in range(1, nc_in + 1):
                            for rc in range(nc):
                                op.append([[1, ja, ra], [2, jb, rb], [0, jc, rc]])
        # AAB operators
        for ra in range(na):
            for rA in range(na):
                for jb in range(1, nb_in):
                    for rb in range(nb):
                        op.append([[1, 0, ra], [1, 1, rA], [2, jb, rb]])
        # AAE operators
        for ra in range(na):
            for rA in range(na):
                for jc in range(1, nc_in + 1):
                    for rc in range(nc):
                        op.append([[1, 0, ra], [1, 1, rA], [0, jc, rc]])
        # ABB operators
        for ja in range(na_in):
            for ra in range(na):
                for jb in range(1, nb):
                    for rB in range(nb):
                        op.append([[1, ja, ra], [2, 0, rb], [2, 1, rB]])
        # BBE operators
        for rb in range(nb):
            for rB in range(nb):
                for jc in range(1, nc_in + 1):
                    for rc in range(nc):
                        op.append([[2, 0, rb], [2, 1, rB], [0, jc, rc]])
        # AEE operators
        for ja in range(na_in):
            for ra in range(na):
                for jc in range(1, nc_in):
                    for jC in range(jc + 1, nc_in + 1):
                        for rc in range(nc):
                            op.append([[1, ja, ra], [0, jc, rc], [0, jC, rc]])
        # BEE operators
        for jb in range(nb_in):
            for rb in range(nb):
                for jc in range(1, nc_in):
                    for jC in range(jc + 1, nc_in + 1):
                        for rc in range(nc):
                            op.append([[2, jb, rb], [0, jc, rc], [0, jC, rc]])

    # Initialize the Gram matrix M
    M = [[None for _ in range(len(op))] for _ in range(len(op))]

    # Fill the Gram matrix
    for ii in range(len(op)):
        for jj in range(ii, len(op)):
            flipped_op_ii = np.flipud(op[ii])
            result = np.vstack((flipped_op_ii, op[jj]))
            M[ii][jj] = simplify_projectors(result)  # ......
            # np.vstack 是用于垂直（行方向）拼接数组的函数。它要求输入的数组在除拼接轴之外的维度上具有相同的形状。
            # 对于二维数组来说，这意味着它们必须具有相同的列数

    # Symmetrize the Gram matrix
    for ii in range(1, len(op)):  # for ii = 2:length(op)
        for jj in range(ii):
            M[ii][jj] = simplify_projectors(np.flipud(M[jj][ii]))

    return M
