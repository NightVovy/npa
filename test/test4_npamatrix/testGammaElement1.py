import numpy as np

# 定义泡利矩阵
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

# 定义角度参数 beta2
beta2 = np.pi / 4  # 例如 45°

# 定义量子操作 A0, A1
A0 = sigma_z
A1 = np.cos(beta2) * sigma_z + np.sin(beta2) * sigma_x

# 计算 A0^dagger 和 A1 的张量积
A0_dagger = A0.conj().T  # A0^dagger 即 A0 的共轭转置
A0_dagger_A1_tensor = np.kron(A0_dagger, A1)  # 计算 A0^dagger 和 A1 的张量积

# 计算 A1^dagger 和 A0 的张量积
A1_dagger = A1.conj().T  # A1^dagger 即 A1 的共轭转置
A1_dagger_A0_tensor = np.kron(A1_dagger, A0)  # 计算 A1^dagger 和 A0 的张量积

# 比较两个张量积是否相等
if np.allclose(A0_dagger_A1_tensor, A1_dagger_A0_tensor):
    print("A0^dagger A1 的张量积与 A1^dagger A0 的张量积相等")
else:
    print("A0^dagger A1 的张量积与 A1^dag量 A0 的张量积不相等")

# 输出结果
print("\nA0^dagger A1 的张量积:\n", A0_dagger_A1_tensor)
print("\nA1^dagger A0 的张量积:\n", A1_dagger_A0_tensor)
