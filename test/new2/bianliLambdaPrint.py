import math
import numpy as np

# 定义步长和范围
step = 0.1
values = np.arange(0.1, 1.1, step)  # 生成从0.1到1.0的值

# lambda_expression 函数定义
def lambda_expression(p00, p01, p10, p11, beta2, theta):
    term1 = math.sqrt((p00 + p10 * math.cos(beta2))**2 + (p10 * math.sin(beta2) * math.sin(2 * theta))**2)
    term2 = math.sqrt((p01 - p11 * math.cos(beta2))**2 + (p11 * math.sin(beta2) * math.sin(2 * theta))**2)

    alpha_term1 = p10**2 / term1
    alpha_term2 = p11**2 / term2

    alpha = (alpha_term1 + alpha_term2) * (math.sin(beta2)**2) * math.cos(2 * theta)

    return term1 + term2 + alpha, alpha

# beta2 的值 (30度, 45度, 60度 转为弧度)
beta2_values = [math.pi / 6, math.pi / 4, math.pi / 3]

# theta 的范围和步长
theta_values = np.arange(0.1, math.pi / 4, 0.1)

# 生成并计算结果
results = []
for p00 in values:
    for p01 in values[values <= p00]:  # 确保p01 <= p00
        for p10 in values[values <= p01]:  # 确保p10 <= p01
            for p11 in values[values <= p10]:  # 确保p11 <= p10
                for beta2 in beta2_values:
                    for theta in theta_values:
                        lambda_result, alpha = lambda_expression(p00, p01, p10, p11, beta2, theta)
                        lhs = math.sqrt((alpha + p00 + p01 + (p10 - p11) * math.cos(beta2))**2 + ((p10 - p11) * math.sin(beta2))**2)
                        lhv = alpha + p00 + p01 + p10 - p11
                        lambdaLessThanLhs = 1 if lambda_result < lhs else 0
                        lambdaLessThanLhv = 1 if lambda_result < lhv else 0
                        lhs_lambda_diff = lhs - lambda_result if lambdaLessThanLhs == 1 else -1
                        lhv_lambda_diff = lhv - lambda_result if lambdaLessThanLhv == 1 else -1
                        results.append((p00, p01, p10, p11, beta2, theta, lambda_result, alpha, lhs, lhv, lambdaLessThanLhs, lambdaLessThanLhv, lhs_lambda_diff, lhv_lambda_diff))

# 将所有结果写入文件
with open("bianliLambda13.txt", "w") as f:
    for combo in results:
        f.write(f"p00={combo[0]:.1f}, p01={combo[1]:.1f}, p10={combo[2]:.1f}, p11={combo[3]:.1f}, beta2={combo[4]:.2f}, theta={combo[5]:.2f}, lambda={combo[6]:.4f}, alpha={combo[7]:.4f}, lhs={combo[8]:.4f}, lhv={combo[9]:.4f}, lambdaLessThanLhs={combo[10]}, lambdaLessThanLhv={combo[11]}, lhs-lambda={combo[12]}, lhv-lambda={combo[13]}\n")

print(f"\nTotal combinations calculated: {len(results)}")
