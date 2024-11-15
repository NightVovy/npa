# 支持向量机
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 读取数据
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = {item.split('=')[0].strip(): float(item.split('=')[1].strip()) for item in line.split(',')}
            data.append(values)
    return pd.DataFrame(data)


# 支持向量机回归拟合
def svr_regression(df, kernel='rbf'):
    X = df[['p00', 'p01']]  # 自变量
    y = df['alpha']  # 因变量
    # 创建 SVR 模型
    model = SVR(kernel=kernel)
    model.fit(X, y)
    return model


# 可视化拟合结果
def plot_svr_fitting(df, model):
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
    ax.set_title('SVR Fit')
    plt.show()


# 运行代码
file_path = 'data3.txt'
data = read_data(file_path)
model_svr = svr_regression(data, kernel='rbf')
plot_svr_fitting(data, model_svr)

# 打印模型支持向量机的相关信息
print(f"SVR model support vectors: {model_svr.support_}")
