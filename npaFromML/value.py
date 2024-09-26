import numpy as np


def value(int_matrix, party):
    tempY = np.array(int_matrix)
    Max = 5
    p = 0
    rowY, colY = tempY.shape
    for ir in range(rowY):
        for jc in range(colY):
            p += tempY[ir, jc] * Max ** (ir * colY + jc)
    posit = np.where(party == p)[0][0]
    # np.where(party == p) 返回一个包含满足条件的索引数组的元组
    # [0] 访问元组中的第一个元素，即索引数组
    # 再加一个 [0] 获取索引数组中的第一个元素，即第一个满足条件的索引
    return posit
