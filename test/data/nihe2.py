import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# 读取数据
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 解析每行数据
            values = {item.split('=')[0].strip(): float(item.split('=')[1].strip()) for item in line.split(',')}
            data.append(values)
    return pd.DataFrame(data)


# 拟合 p00, p01 与 alpha 的关系
def fit_data(df):
    X = df[['p00', 'p01']]  # 自变量
    y = df['alpha']  # 因变量
    model = LinearRegression()
    model.fit(X, y)
    return model


# 可视化拟合结果
def plot_fitting(df, model):
    # 生成网格数据
    p00_range = np.linspace(df['p00'].min(), df['p00'].max(), 100)
    p01_range = np.linspace(df['p01'].min(), df['p01'].max(), 100)
    p00_grid, p01_grid = np.meshgrid(p00_range, p01_range)
    X_grid = np.c_[p00_grid.ravel(), p01_grid.ravel()]

    # 预测 alpha
    alpha_grid = model.predict(X_grid).reshape(p00_grid.shape)

    # 绘制 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(p00_grid, p01_grid, alpha_grid, cmap='viridis')
    ax.set_xlabel('p00')
    ax.set_ylabel('p01')
    ax.set_zlabel('alpha')
    ax.set_title('Fitting p00, p01 to alpha')
    plt.show()


# 运行代码
file_path = 'data3.txt'
data = read_data(file_path)
model = fit_data(data)
plot_fitting(data, model)

# 打印模型参数
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")
