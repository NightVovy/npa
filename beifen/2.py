import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 目标函数：最小化第二个公式的平方
def objective(params):
    p00, p01, p10, p11, beta2, theta = params
    left_2 = left_side_2(beta2, p00, p01, p10, p11, theta)
    return left_2 ** 2  # 最小化平方


# 计算第二个公式左侧
def left_side_2(beta2, p00, p01, p10, p11, theta):
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    term1 = p10 * np.sin(beta2) * (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    term2 = -p11 * np.sin(beta2) * (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return term1 + term2


# 计算 sin(mu1), sin(mu2), cos(mu1), cos(mu2)
def compute_trig_functions(beta2, p00, p01, p10, p11, theta):
    cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    sin_mu2 = - (p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return cos_mu1, cos_mu2, sin_mu1, sin_mu2


# 运行多次优化，取消约束
def multi_start_optimization_no_constraints_fixed_params(n_trials=30, beta2=np.pi / 4, theta=np.pi / 8):
    solutions = []

    for _ in range(n_trials):
        # 随机初始参数，确保在 (0, 1) 之间
        initial_params = [
            np.random.uniform(1e-5, 1 - 1e-5),  # p00
            np.random.uniform(1e-5, 1 - 1e-5),  # p01
            np.random.uniform(1e-5, 1 - 1e-5),  # p10
            np.random.uniform(1e-5, 1 - 1e-5),  # p11
            np.random.uniform(1e-5, np.pi / 2 - 1e-5),  # beta2
            np.random.uniform(1e-5, np.pi / 4 - 1e-5)  # theta
        ]

        # 变量边界
        bounds = [
            (1e-5, 1 - 1e-5),  # p00 in (0, 1)
            (1e-5, 1 - 1e-5),  # p01 in (0, 1)
            (1e-5, 1 - 1e-5),  # p10 in (0, 1)
            (1e-5, 1 - 1e-5),  # p11 in (0, 1)
            (1e-5, np.pi / 2 - 1e-5),  # beta2 in (0, pi/2)
            (1e-5, np.pi / 4 - 1e-5)  # theta in (0, pi/4)
        ]

        # 执行优化（无约束）
        result = minimize(objective, initial_params, args=(), bounds=bounds)

        # 只保存成功收敛的解
        if result.success:
            solutions.append((result.x, result.fun))

    return solutions


# 运行多次优化，取消约束
multi_solutions_no_constraints_fixed = multi_start_optimization_no_constraints_fixed_params(n_trials=30)

# 显示所有找到的解
results_no_constraints_fixed = []
for params, value in multi_solutions_no_constraints_fixed:
    p00, p01, p10, p11, beta2, theta = params
    results_no_constraints_fixed.append({
        'p00': p00,
        'p01': p01,
        'p10': p10,
        'p11': p11,
        'beta2 (rad)': beta2,
        'theta (rad)': theta,
        'Minimized Value': value
    })


# 筛选掉不符合条件的数据（极端值）
def filter_extreme_results(results):
    filtered_results = []

    # 设置筛选条件：p0, p1, p10, p11 要在 [0.05, 0.95] 之间，beta2 和 theta 要大于 0.1
    for result in results:
        p00, p01, p10, p11, beta2, theta, minimized_value = result.values()

        # 判断是否符合筛选条件
        if (0.05 <= p00 <= 0.95 and 0.05 <= p01 <= 0.95 and 0.05 <= p10 <= 0.95 and 0.05 <= p11 <= 0.95 and
                beta2 > 0.1 and theta > 0.1):
            filtered_results.append(result)

    return filtered_results


# 筛选结果
filtered_results = filter_extreme_results(results_no_constraints_fixed)

# 转换为 DataFrame 并准备进行拟合
filtered_data = pd.DataFrame(filtered_results)

# 提取特征（beta2, theta）和目标（p00, p01, p10, p11）
X = filtered_data[['beta2 (rad)', 'theta (rad)']]  # 特征：beta2 和 theta
y = filtered_data[['p00', 'p01', 'p10', 'p11']]  # 目标：p00, p01, p10, p11

# 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 输出拟合后的系数（即beta2和theta与p00, p01, p10, p11之间的关系）
print("Coefficients for p00, p01, p10, p11 with respect to beta2 and theta:", model.coef_)
