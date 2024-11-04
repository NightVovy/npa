import numpy as np


# 计算 A 和 C
def compute_A_C(beta, p00, p10, p11, p01):
    A = p00 + p10 * np.cos(beta)
    C = p01 - p11 * np.cos(beta)
    return A, C


# 计算 beta
def compute_beta(theta, p00, p01, p10, p11):
    sin_theta = np.sin(theta)
    sin_2theta = np.sin(2 * theta)

    for beta in np.linspace(0, np.pi / 2, 1000):
        A, C = compute_A_C(beta, p00, p10, p11, p01)

        if C ** 2 - A ** 2 <= 0:  # 确保分母不为零
            continue

        left_side = (np.sin(beta)) ** 2 / (sin_2theta) ** 2
        right_side = (p10 ** 2 - p11 ** 2) * (A ** 2) * (C ** 2) / (p10 ** 2 * p11 ** 2 * (C ** 2 - A ** 2))

        if right_side <= 0:  # 检查右侧是否有效
            continue

        if np.isclose(left_side, right_side, atol=1e-6):
            return beta
    raise ValueError("未找到有效的 beta 值")


# 计算 alpha
def compute_alpha(beta, theta, p00, p01, p10, p11):
    sin_beta = np.sin(beta)
    sin_2theta = np.sin(2 * theta)

    A, C = compute_A_C(beta, p00, p10, p11, p01)
    B = p01 - p10 * np.cos(beta)

    if B <= 0 or A + p10 ** 2 * sin_beta ** 2 * sin_2theta ** 2 <= 0:
        return None  # 返回 None 以表示无效值

    alpha = (p10 ** 2 * sin_beta * sin_2theta / np.sqrt(A + p10 ** 2 * sin_beta ** 2 * sin_2theta ** 2) +
             p11 ** 2 * sin_beta * sin_2theta / np.sqrt(B + p11 ** 2 * sin_beta ** 2 * sin_2theta ** 2)) * (
                        sin_beta * np.cos(2 * theta) / sin_2theta)

    return alpha


# 设置参数范围和步长
step = 0.1  # 调整步长以增加精度
theta_initial = 0.1  # 初始值，接近 0
theta_step = 0.05  # theta 的步长

# 遍历 p00, p01, p10, p11
results = []

for p00 in np.arange(0, 1.1, step):
    for p01 in np.arange(0, 1.1, step):
        for p10 in np.arange(0, 1.1, step):
            for p11 in np.arange(0, 1.1, step):
                if p10 == p11:
                    continue  # p10 不等于 p11

                # 遍历 theta
                theta = theta_initial
                while theta < np.pi / 4:
                    try:
                        beta = compute_beta(theta, p00, p01, p10, p11)
                        alpha = compute_alpha(beta, theta, p00, p01, p10, p11)
                        if alpha is not None:
                            results.append((p00, p01, p10, p11, alpha, beta, theta))
                    except ValueError:
                        pass  # 忽略找不到解的情况

                    theta += theta_step

# 输出所有结果
for result in results:
    p00, p01, p10, p11, alpha, beta, theta = result
    print(
        f"p00: {p00:.1f}, p01: {p01:.1f}, p10: {p10:.1f}, p11: {p11:.1f}, alpha: {alpha:.6f}, beta: {beta:.6f}, theta: {theta:.6f}")
