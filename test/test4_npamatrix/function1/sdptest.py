import cvxpy as cp
from collections import defaultdict
from stringtest2 import generate_string_sets
from matrixtest2 import create_upper_triangle_matrix, record_matrix_elements, print_matrix


def get_string_set_and_matrix(S1, layer):
    # 通过 generate_string_sets 获取字符串集合
    S = generate_string_sets(layer, S1)
    print(f"Generated String Set (S{layer}): {S}")

    # 创建上三角矩阵
    matrix = create_upper_triangle_matrix(S)
    print(f"S{layer} Upper Triangle Matrix:")
    print_matrix(matrix)  # 调用 matrixtest2.py 中的 print_matrix 函数打印矩阵

    # 记录矩阵元素的出现次数和位置
    element_info = record_matrix_elements(matrix)
    return S, element_info, matrix


def construct_sdp_constraints(element_info, n):
    # 初始化gamma矩阵，大小为n x n，所有元素为符号变量
    gamma = cp.Variable((n, n))

    # 添加约束条件
    constraints = []

    # 1. 对角线元素为1
    # for i in range(n):
    #     constraints.append(gamma[i, i] == 1)
    constraints.append(cp.diag(gamma) == 1)

    # 2. 半正定矩阵约束：gamma是半正定的
    constraints.append(gamma >> 0)

    # 3. Hermitian symmetry：gamma是厄米矩阵
    # constraints.append(gamma == gamma.H)
    constraints.append(gamma == cp.conj(gamma.T))

    # 4. 构造基于记录的约束：对于每个出现次数大于1的元素，根据其出现的位置添加约束
    for element, info in element_info.items():
        positions = info['positions']
        count = info['count']
        if count > 1:
            for i in range(count):
                for j in range(i + 1, count):
                    pos_i = positions[i]
                    pos_j = positions[j]
                    constraints.append(gamma[pos_i[0], pos_i[1]] == gamma[pos_j[0], pos_j[1]])

    return gamma, constraints


def construct_objective(gamma):
    # 目标函数为：gamma[1, 3] + gamma[1, 4] + gamma[2, 3] - gamma[2, 4]
    return cp.Maximize(gamma[1, 3] + gamma[1, 4] + gamma[2, 3] - gamma[2, 4])


def solve_sdp_problem(S1, layer):
    # 获取字符串集合、元素信息和矩阵
    S, element_info, matrix = get_string_set_and_matrix(S1, layer)

    # 检查 element_info 是否为 None
    if element_info is None:
        print("Error: element_info is None")
        return

    # 字符串集合的大小
    n = len(S)

    # 构造SDP约束条件
    gamma, constraints = construct_sdp_constraints(element_info, n)

    # 构造目标函数
    objective = construct_objective(gamma)

    # 设置SDP优化问题
    problem = cp.Problem(objective, constraints)

    # 求解SDP问题
    problem.solve(solver="SDPA")

    # 输出求解结果
    print(f"Optimal value: {problem.value}")
    print("Optimal gamma matrix:")
    print(gamma.value)

    # 输出gamma[1, 3], gamma[1, 4], gamma[2, 3], gamma[2, 4]
    print(f"gamma[1, 3]: {gamma.value[1, 3]}")
    print(f"gamma[1, 4]: {gamma.value[1, 4]}")
    print(f"gamma[2, 3]: {gamma.value[2, 3]}")
    print(f"gamma[2, 4]: {gamma.value[2, 4]}")


# 示例输入
S1 = {"I", "A0", "A1", "B0", "B1"}
layer = 2  # 可以修改层数来生成不同的字符串集合

# 调用函数来求解SDP问题
solve_sdp_problem(S1, layer)
