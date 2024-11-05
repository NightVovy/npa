import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
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

# 使用XGBoost和LightGBM拟合参数
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

    # XGBoost 模型训练
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42)
    xgb_model.fit(X_scaled, alpha_values)
    xgb_predicted_alpha = xgb_model.predict(X_scaled)

    # LightGBM 模型训练
    lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=200, random_state=42)
    lgb_model.fit(X_scaled, alpha_values)
    lgb_predicted_alpha = lgb_model.predict(X_scaled)

    # 可视化拟合效果（XGBoost）
    plt.figure(figsize=(10, 6))
    plt.scatter(alpha_values, xgb_predicted_alpha, color='blue', label='XGBoost Predicted vs Actual')
    plt.plot([min(alpha_values), max(alpha_values)], [min(alpha_values), max(alpha_values)], color='red', linestyle='--', label='Perfect fit')
    plt.xlabel('Actual alpha')
    plt.ylabel('Predicted alpha')
    plt.title('XGBoost Fit of alpha vs p00, p01, p10 (without p11)')
    plt.legend()
    plt.show()

    # 可视化拟合效果（LightGBM）
    plt.figure(figsize=(10, 6))
    plt.scatter(alpha_values, lgb_predicted_alpha, color='green', label='LightGBM Predicted vs Actual')
    plt.plot([min(alpha_values), max(alpha_values)], [min(alpha_values), max(alpha_values)], color='red', linestyle='--', label='Perfect fit')
    plt.xlabel('Actual alpha')
    plt.ylabel('Predicted alpha')
    plt.title('LightGBM Fit of alpha vs p00, p01, p10 (without p11)')
    plt.legend()
    plt.show()

    return xgb_model, lgb_model

# 读取数据
data = read_data('data.txt')

# 使用XGBoost和LightGBM模型拟合参数
xgb_model, lgb_model = fit_parameters_with_advanced_models(data)
