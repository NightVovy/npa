import numpy as np

# 定义泡利矩阵
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

# 自定义角度
beta2 = np.pi / 4  # 比如 45°
mu1 = np.pi / 6    # 比如 30°
mu2 = np.pi / 3    # 比如 60°

# 定义量子操作 A0, A1, B0
A0 = sigma_z
A1 = np.cos(beta2) * sigma_z + np.sin(beta2) * sigma_x
B0 = np.cos(mu1) * sigma_z + np.sin(mu1) * sigma_x
B1 = np.cos(mu2) * sigma_z + np.sin(mu2) * sigma_x

# 计算 A0^dagger 和 A1 B0 的乘积
A0_dagger = A0.conj().T  # A0^dagger 即 A0 的共轭转置
A0_dagger_A1_B0_product = np.dot(A0_dagger, np.dot(A1, B0))  # A0^dagger * A1 * B0 的乘积

# 计算 A1^dagger 和 A0 B0 的乘积
A1_dagger = A1.conj().T  # A1^dagger 即 A1 的共轭转置
A1_dagger_A0_B0_product = np.dot(A1_dagger, np.dot(A0, B0))  # A1^dagger * A0 * B0 的乘积

# 计算 A0^dagger 和 A0 的乘积
A0_dagger_A0_product = np.dot(A0_dagger, A0)

# 比较两个矩阵乘积是否相等
if np.allclose(A0_dagger_A1_B0_product, A1_dagger_A0_B0_product):
    print("A0^dagger A1 B0 的乘积与 A1^dagger A0 B0 的乘积相等")
else:
    print("A0^dagger A1 B0 的乘积与 A1^dagger A0 B0 的乘积不相等")

# 输出 A0^dagger 和 A0 的乘积结果
print("\nA0^dagger A0 的乘积:\n", A0_dagger_A0_product)

# 输出其他结果
print("\nA0^dagger A1 B0 的矩阵乘积:\n", A0_dagger_A1_B0_product)
print("\nA1^dagger A0 B0 的矩阵乘积:\n", A1_dagger_A0_B0_product)
