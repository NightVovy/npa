def generate_text(p, q, alpha, layer):
    # 根据 p 和 q 计算 p00, p01, p10, p11，并保留2位小数
    p00 = round(p * q, 2)
    p01 = round(p * (1 - q), 2)
    p10 = round((1 - p) * q, 2)
    p11 = round((1 - p) * (1 - q), 2)

    # 根据 alpha、p00、p01、p10、p11 和 layer 生成最终的文字
    text = f'npa_max({alpha:.2f} * A1 + {p00:.2f} * A1 * B1 + {p01:.2f} * A1 * B2 + {p10:.2f} * A2 * B1 - {p11:.2f} * A2 * B2, "{layer}")'

    return text


def generate_second_text(p00_input, p01_input, p10_input, p11_input, alpha, layer2):
    # 根据输入的 p00_input, p01_input, p10_input, p11_input 和 alpha 生成新的文字
    text = f'npa_max({alpha:.2f} * A1 + {p00_input} * A1 * B1 + {p01_input} * A1 * B2 + {p10_input} * A2 * B1 - {p11_input} * A2 * B2, "{layer2}")'

    return text


# 在函数外部设置 p, q, alpha 和 layer 的值
p = 0.7
q = 0.5
alpha = 1.5
layer = 3  # 如果需要修改 layer 的值

# 调用第一个函数生成并输出结果
result_1 = generate_text(p, q, alpha, layer)
print(result_1)

# 调用第二个函数生成并输出结果
p00_input = 0.9
p01_input = 0.9
p10_input = 1
p11_input = 1
alpha2 = 0.4
layer2 = "1 + A B + A^2 B"  # 如果需要修改 layer 的值
result_2 = generate_second_text(p00_input, p01_input, p10_input, p11_input, alpha2, layer2)
print(result_2)