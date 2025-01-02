import numpy as np
import re


# 使用 numpy.isclose() 进行浮动数比较，允许一定的误差范围
def is_approximately_equal(a, b, tol=1e-6):
    return np.isclose(a, b, atol=tol)


# 打开文件并读取内容
def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            # 去除行尾空白
            line = line.strip()

            # 调试输出每一行
            print(f"Reading line: {repr(line)}")  # 输出带有空格和特殊字符的原始行

            # 使用正则表达式提取参数和结果
            match = re.match(
                r'alpha=([0-9\.]+),\s*p00=([0-9\.]+),\s*p01=([0-9\.]+),\s*p10=([0-9\.]+),\s*p11=([0-9\.]+)\.\s*result\s*=\s*([0-9\.]+)',
                line)

            if match:
                # 提取到的所有参数
                alpha = float(match.group(1))
                p00 = float(match.group(2))
                p01 = float(match.group(3))
                p10 = float(match.group(4))
                p11 = float(match.group(5))
                result = float(match.group(6))

                # 输出提取到的每个参数
                print(f"Extracted alpha = {alpha}")
                print(f"Extracted p00 = {p00}")
                print(f"Extracted p01 = {p01}")
                print(f"Extracted p10 = {p10}")
                print(f"Extracted p11 = {p11}")
                print(f"Extracted result = {result}")

                # 跳过 result 为 99999 的行
                if result == 99999:
                    print(f"Skipping line due to result=99999: {line}")
                    continue

                # 将提取到的数据存储到列表中
                params = {
                    'alpha': alpha,
                    'p00': p00,
                    'p01': p01,
                    'p10': p10,
                    'p11': p11
                }
                data.append((params, result))
            else:
                print(f"Skipping line due to format mismatch: {line}")
    return data


def compare_results(lilunzhi_file, nparesult_file):
    # 读取两个文件的内容
    lilunzhi_data = read_file(lilunzhi_file)
    npa_data = read_file(nparesult_file)

    # 将 npa_data 按照 alpha, p00, p01, p10, p11 作为键存储方便查找
    npa_dict = {}
    for params, result in npa_data:
        key = (params['alpha'], params['p00'], params['p01'], params['p10'], params['p11'])
        npa_dict[key] = result

    # 比较两个文件中的结果
    output = []
    for params, lilunzhi in lilunzhi_data:
        key = (params['alpha'], params['p00'], params['p01'], params['p10'], params['p11'])

        # 查找相同的参数组合，忽略微小差异
        for npa_key, npa in npa_dict.items():
            if is_approximately_equal(params['alpha'], npa_key[0]) and \
                    is_approximately_equal(params['p00'], npa_key[1]) and \
                    is_approximately_equal(params['p01'], npa_key[2]) and \
                    is_approximately_equal(params['p10'], npa_key[3]) and \
                    is_approximately_equal(params['p11'], npa_key[4]):

                # 检查lilunzhi是否大于npa且不为99999
                if lilunzhi != 99999 and lilunzhi > npa:
                    # 计算差值并保存结果
                    difference = lilunzhi - npa
                    output.append(
                        f"alpha={params['alpha']}, p00={params['p00']}, p01={params['p01']}, p10={params['p10']}, p11={params['p11']}, lilunzhi={lilunzhi}, npa={npa}, difference={difference}")

    return output


# 运行比较函数并打印输出
result = compare_results("lilunBianlip10p111.txt", "nparesultp10p111.txt")
if result:
    for line in result:
        print(line)
else:
    print("没有找到满足条件的记录。")
