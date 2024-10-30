# 读取文件并提取lambda=4.00000的数据
output_lines = []

with open('data.txt', 'r') as file:
    for line in file:
        if 'lambda=4.00000' in line:
            output_lines.append(line.strip())

# 输出符合条件的结果
for output in output_lines:
    print(output)
