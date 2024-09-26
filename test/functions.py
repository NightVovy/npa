import math

import numpy as np


def measure_pure_state(psi, A0, B0):
    # 计算A0和B0的张量积
    operator = np.kron(A0, B0)

    # 计算量子态psi的共轭转置
    psi_dagger = psi.conj().T

    # 计算测量结果
    result = psi_dagger @ operator @ psi

    return result


def sort_numbers_with_names(a, b, c):
    numbers = {'实际值': a, '理论值': b, '反推': c}
    sorted_numbers = sorted(numbers.items(), key=lambda item: item[1])
    return sorted_numbers
