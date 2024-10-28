import re
import pandas as pd
from sklearn.linear_model import LinearRegression

# 读取文件内容
with open('data.txt', 'r') as file:
    content = file.read()

# 提取数据
pattern = r"p00=(.*?), p01=(.*?), p10=(.*?), p11=(.*?), alpha=(.*?), cosbeta2=(.*?), lambda=(.*?), problem.value=(.*?)"
matches = re.findall(pattern, content)

# 转换为DataFrame
data = pd.DataFrame(matches, columns=['p00', 'p01', 'p10', 'p11', 'alpha', 'cosbeta2', 'lambda', 'problem_value'])
data = data.astype(float)

# 回归分析
X = data[['p00', 'p01', 'p10', 'p11', 'cosbeta2', 'problem_value']]
y = data['lambda']
model = LinearRegression()
model.fit(X, y)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
