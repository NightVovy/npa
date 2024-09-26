import numpy as np


def simplify_projectors(in_matrix):
    """
    辅助函数，用于生成 NPA 矩阵。

    参数:
    in_matrix (numpy.ndarray): 一个 n x 3 的矩阵，每行表示 [party, input, output]。
    不是把M塞进去
    -- 不同party的算符对易
    -- 假设投影算符Ai_0Aj_0 = delta_ij Ai_0
    identity is denoted as [0 0 0]
    and zero is denoted as [-1 -1 -1]

    返回:
    numpy.ndarray: 简化后的矩阵。
    """
    out = []
    in_matrix = in_matrix[in_matrix[:, 0].argsort()]  # 按照第一列排序
    # in_matrix[:, 0]：选择 in_matrix 的第一列
    # argsort()：返回排序后的索引
    # in_matrix[in_matrix[:, 0].argsort()]：使用排序后的索引重新排列 in_matrix 的行

    # 找到不同的 party
    parties = np.unique(in_matrix[:, 0])

    my_flag = False
    for party in parties:
        tmp1 = in_matrix[in_matrix[:, 0] == party]  # 返回符合当前party的行

        while True:
            tmp2 = tmp1.copy()
            for ii in range(len(tmp2) - 1):
                if tmp2[ii, 1] == tmp2[ii + 1, 1]:
                    if tmp2[ii, 2] == tmp2[ii + 1, 2]:
                        # 相同party相同输入输出
                        tmp2[ii + 1, :] = [0, 0, 0]
                    else:
                        # 相同party相同输入不同输出 A1_0A0_0 = 0; zero = [-1 -1 -1];
                        tmp2 = np.array([[-1, -1, -1]])
                        my_flag = True
                        break

            if len(tmp2) == len(tmp1):
                break
            # 如果 tmp2 的行数与 tmp1 相同，则跳出 while 循环
            # 否则，将 tmp2 赋值给 tmp1，继续下一轮循环
            tmp1 = tmp2

        if my_flag:
            out = np.array([[-1, -1, -1]])
            break

        out.append(tmp2)

    out = np.vstack(out)  # 为什么out还要拼在out后面？ 因为out之前是个列表？
    out = out[out[:, 0].argsort()]

    # 移除单位算子 [0, 0, 0]
    out = out[np.sum(out, axis=1) != 0]
    # 使用布尔数组作为索引，选择 out 数组中和不为零的行

    if out.size == 0:
        out = np.array([[0, 0, 0]])

    return out
