import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: 读取文件并提取 p 和 theta 的值
p_values = []
theta_values = []

# 假设文件路径为 'data2.txt'，你可以根据需要修改文件路径
with open('data2.txt', 'r') as file:
    for line in file:
        # 删除行末的换行符
        line = line.strip()

        # 按照逗号分隔每一行的内容
        parts = line.split(', ')

        # 提取 p 和 theta 的值
        p_value = float(parts[0].split('=')[1])  # 获取 p 的值
        theta_value = float(parts[2].split('=')[1])  # 获取 theta 的值

        # 添加到列表中
        p_values.append(p_value)
        theta_values.append(theta_value)

# Step 2: 将 p_values 和 theta_values 转换为 numpy 数组
p_values = np.array(p_values).reshape(-1, 1)
theta_values = np.array(theta_values).reshape(-1, 1)

# Step 3: 创建并训练线性回归模型
model = LinearRegression()
model.fit(p_values, theta_values)

# Step 4: 获取回归系数
a = model.coef_[0][0]  # 斜率
b = model.intercept_[0]  # 截距

# 打印回归系数
print(f"回归系数 (a): {a}")
print(f"截距 (b): {b}")

# Step 5: 可视化结果
plt.figure(figsize=(8, 6))

# 绘制散点图
plt.scatter(p_values, theta_values, color='blue', label='数据点')

# 绘制回归线
plt.plot(p_values, model.predict(p_values), color='red', label=f'回归线: θ = {a:.2f}p + {b:.2f}')

# 设置图表标签和标题
plt.xlabel('p 值')
plt.ylabel('θ 值')
plt.title('p 和 θ 之间的关系')
plt.legend()

# 显示图表
plt.show()
