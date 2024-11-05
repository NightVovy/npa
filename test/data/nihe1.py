import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 读取数据
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                params = {}
                parts = line.split(',')
                for part in parts:
                    key, value = part.split('=')
                    params[key.strip()] = float(value.strip())
                data.append(params)
    return data

# 使用随机森林和 SVR 模型拟合参数
def fit_parameters_with_advanced_models(data):
    # 提取需要拟合的变量，去掉 p11
    p00 = np.array([entry['p00'] for entry in data])
    p01 = np.array([entry['p01'] for entry in data])
    p10 = np.array([entry['p10'] for entry in data])
    alpha_values = np.array([entry['alpha'] for entry in data])

    # 准备特征矩阵，并添加更高阶交互特征
    X = np.column_stack([p00, p01, p10,
                         p00*p01, p01*p10, p00*p10,  # 二次交互项
                         p00**2, p01**2, p10**2,  # 二次自交互项
                         p00*p01*p10, p00**3, p01**3, p10**3])  # 三次交互项和自交互项

    # 标准化特征（可以试试 MinMaxScaler 来替代）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用随机森林回归进行拟合
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_scaled, alpha_values)
    rf_predicted_alpha = rf_model.predict(X_scaled)

    # 使用支持向量机回归 (SVR) 进行拟合，并加上多项式特征（degree=4）
    svr_model = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=4), SVR(kernel='rbf', C=100, epsilon=0.1))
    svr_model.fit(X_scaled, alpha_values)
    svr_predicted_alpha = svr_model.predict(X_scaled)

    # 可视化拟合效果（随机森林）
    plt.figure(figsize=(10, 6))
    plt.scatter(alpha_values, rf_predicted_alpha, color='blue', label='RF Predicted vs Actual')
    plt.plot([min(alpha_values), max(alpha_values)], [min(alpha_values), max(alpha_values)], color='red', linestyle='--', label='Perfect fit')
    plt.xlabel('Actual alpha')
    plt.ylabel('Predicted alpha')
    plt.title('Random Forest Fit of alpha vs p00, p01, p10 (without p11)')
    plt.legend()
    plt.show()

    # 可视化拟合效果（SVR）
    plt.figure(figsize=(10, 6))
    plt.scatter(alpha_values, svr_predicted_alpha, color='green', label='SVR Predicted vs Actual')
    plt.plot([min(alpha_values), max(alpha_values)], [min(alpha_values), max(alpha_values)], color='red', linestyle='--', label='Perfect fit')
    plt.xlabel('Actual alpha')
    plt.ylabel('Predicted alpha')
    plt.title('SVR Fit of alpha vs p00, p01, p10 (without p11)')
    plt.legend()
    plt.show()

    return rf_model, svr_model

# 读取数据
data = read_data('data.txt')

# 使用随机森林和 SVR 模型拟合参数
rf_model, svr_model = fit_parameters_with_advanced_models(data)
