import math


def calculate_formula(alpha, p00, p01, p10, p11, beta):
    # 计算公式的值
    cos_beta = math.cos(beta)
    sin_beta = math.sin(beta)

    term1 = alpha + p00 + p01
    term2 = (p10 - p11) * cos_beta
    term3 = (p10 - p11) * sin_beta

    result = math.sqrt((term1 + term2) ** 2 + term3 ** 2)
    return result


# 示例：替换这些值为您实际的输入
alpha = 0.2
p00 = 0.4
p01 = 0.9
p10 = 0.8
p11 = 0.4
beta = 1.56769674324185

# 计算结果
result = calculate_formula(alpha, p00, p01, p10, p11, beta)
print("计算结果为:", result)
