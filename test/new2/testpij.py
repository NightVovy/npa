import numpy as np

# 定义步长和范围
step = 0.1
values = np.arange(0.1, 1.1, step)  # 生成从0.1到1.0的值

# 生成并打印满足条件的组合
result = []
for p00 in values:
    for p01 in values[values <= p00]:  # 确保p01 <= p00
        for p10 in values[values <= p01]:  # 确保p10 <= p01
            for p11 in values[values <= p10]:  # 确保p11 <= p10
                result.append((p00, p01, p10, p11))

# 打印结果
for combo in result:
    print(combo)

print(f"\nTotal combinations: {len(result)}")
