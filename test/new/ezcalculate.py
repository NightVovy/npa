import math


# 计算 IQ 的函数，接受 p00, p01, p10, p11
def calculate_IQ_from_p_values(p00, p01, p10, p11):
    # 计算第一部分：sqrt(p00 * p10 + p01 * p11)
    part1 = math.sqrt(p00 * p10 + p01 * p11)

    # 计算第二部分：sqrt((p00^2 + p10^2) / (p00 * p10) + (p01^2 + p11^2) / (p01 * p11))
    part2 = math.sqrt((p00 ** 2 + p10 ** 2) / (p00 * p10) + (p01 ** 2 + p11 ** 2) / (p01 * p11))

    # 计算最终的 IQ 值
    IQ = part1 * part2

    return IQ


# 计算 p00, p01, p10, p11 的函数（基于 p 和 q）
def calculate_p_values_from_pq(p, q):
    # 根据 p 和 q 计算 p00, p01, p10, p11
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    return p00, p01, p10, p11


# 计算 example2 的函数，基于 alpha 和 beta
def calculate_example2(alpha, beta):
    # 计算 sqrt((4 + alpha^2) * (1 + beta^2))
    example2 = math.sqrt((4 + alpha ** 2) * (1 + beta ** 2))
    return example2


# 计算 sqrt(8 + 2 * alpha^2) 的函数
def calculate_example3(alpha):
    # 计算 sqrt(8 + 2 * alpha^2)
    example3 = math.sqrt(8 + 2 * alpha**2)
    return example3


# 示例 1: 使用 p 和 q 计算 p00, p01, p10, p11
p = 0.6
q = 0.7
p00, p01, p10, p11 = calculate_p_values_from_pq(p, q)
IQ_from_pq = calculate_IQ_from_p_values(p00, p01, p10, p11)
print(f"The value of I_Q (calculated from p and q) is: {IQ_from_pq}")

# 示例 2: 直接输入 p00, p01, p10, p11
p00_input = 0.42
p01_input = 0.18
p10_input = 0.28
p11_input = 0.12
IQ_from_inputs = calculate_IQ_from_p_values(p00_input, p01_input, p10_input, p11_input)
print(f"The value of I_Q (calculated from direct p-values) is: {IQ_from_inputs}")

# 示例 3: 输入 alpha 和 beta 来计算 example2
alpha = 1.3  # 输入 alpha 的值
beta = 0.8  # 输入 beta 的值
example2_value = calculate_example2(alpha, beta)
print(f"The value of example2 (sqrt((4 + alpha^2) * (1 + beta^2))) is: {example2_value}")

# 输出提示信息
print(f"Note: p00 = p01 = {beta}, p10 = p11 = 1")


# 示例 4: 输入 alpha 来计算 example3 (sqrt(8 + 2 * alpha^2))
example3_value = calculate_example3(alpha)
print(f"The value of example3 (sqrt(8 + 2 * alpha^2)) is: {example3_value}")

# 输出提示信息
print(f"Note: p_ij = 1")