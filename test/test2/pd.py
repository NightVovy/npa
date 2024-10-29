import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 从文件中读取数据
data = []
with open('data.txt', 'r') as file:
    for line in file:
        data_dict = {key.strip(): float(value.strip()) for key, value in (item.split('=') for item in line.split(','))}
        data.append(data_dict)

# 转换为DataFrame
df = pd.DataFrame(data)

# 准备特征和目标变量
X = df[['p00', 'p01', 'p10', 'p11', 'alpha']]  # 特征
y = df['lambda']  # 目标变量

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
model = RandomForestRegressor(random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# 可视化真实值与预测值
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 45-degree line
plt.show()
