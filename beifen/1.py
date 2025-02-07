# 多次随机初始化优化，取消 p00 >= p01 >= p10 >= p11 约束
def multi_start_optimization_no_constraints(n_trials=10):
    solutions = []

    for _ in range(n_trials):
        # 随机初始参数，确保在 (0, 1) 之间
        initial_params = [
            np.random.uniform(1e-5, 1 - 1e-5),  # p00
            np.random.uniform(1e-5, 1 - 1e-5),  # p01
            np.random.uniform(1e-5, 1 - 1e-5),  # p10
            np.random.uniform(1e-5, 1 - 1e-5),  # p11
            np.random.uniform(1e-5, np.pi / 2 - 1e-5),  # beta2
            np.random.uniform(1e-5, np.pi / 4 - 1e-5)   # theta
        ]

        # 变量边界
        bounds = [
            (1e-5, 1 - 1e-5),  # p00 in (0, 1)
            (1e-5, 1 - 1e-5),  # p01 in (0, 1)
            (1e-5, 1 - 1e-5),  # p10 in (0, 1)
            (1e-5, 1 - 1e-5),  # p11 in (0, 1)
            (1e-5, np.pi / 2 - 1e-5),  # beta2 in (0, pi/2)
            (1e-5, np.pi / 4 - 1e-5)   # theta in (0, pi/4)
        ]

        # 执行优化（无约束）
        result = minimize(objective, initial_params, bounds=bounds)

        # 只保存成功收敛的解
        if result.success:
            solutions.append((result.x, result.fun))

    return solutions

# 运行多次优化，取消约束
multi_solutions_no_constraints = multi_start_optimization_no_constraints(n_trials=10)

# 显示所有找到的解
results_no_constraints = []
for params, value in multi_solutions_no_constraints:
    p00, p01, p10, p11, beta2, theta = params
    results_no_constraints.append({
        'p00': p00,
        'p01': p01,
        'p10': p10,
        'p11': p11,
        'beta2 (rad)': beta2,
        'theta (rad)': theta,
        'Minimized Value': value
    })

# 转换为 DataFrame 并展示
df_results_no_constraints = pd.DataFrame(results_no_constraints)
tools.display_dataframe_to_user("Optimized Solutions without Constraints", df_results_no_constraints)
