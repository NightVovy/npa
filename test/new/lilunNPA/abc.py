import re


# 用于比较两个浮点数是否近似相等
def is_approximately_equal(val1, val2, tolerance=1e-9):
    return abs(val1 - val2) <= tolerance


def read_file(filename):
    """
    读取文件并提取每一行的参数和result值，确保正确提取
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # 使用正则表达式来提取参数和值，确保即使有空格也能正确处理
            match = re.match(
                r'alpha=([0-9\.]+),\s*p00=([0-9\.]+),\s*p01=([0-9\.]+),\s*p10=([0-9\.]+),\s*p11=([0-9\.]+)\.\s*result\s*=\s*([0-9\.]+)',
                line.strip())
            if match:
                # 提取参数值并转化为浮点数
                params = {
                    'alpha': float(match.group(1)),
                    'p00': float(match.group(2)),
                    'p01': float(match.group(3)),
                    'p10': float(match.group(4)),
                    'p11': float(match.group(5)),
                }
                result = float(match.group(6))
                data.append((params, result))
                # 调试：输出提取的参数和值
                print(
                    f"Extracted alpha = {params['alpha']}, p00 = {params['p00']}, p01 = {params['p01']}, p10 = {params['p10']}, p11 = {params['p11']}, result = {result}")
            else:
                print(f"Skipping line due to format mismatch: {line.strip()}")
    return data


def compare_results(lilunzhi_file, nparesult_file, output_file):
    """
    比较两个文件中的结果，并将符合条件的结果写入输出文件
    """
    # 读取两个文件的数据
    lilunzhi_data = read_file(lilunzhi_file)
    nparesult_data = read_file(nparesult_file)

    # 打开输出文件，准备写入符合条件的结果
    with open(output_file, 'w') as output:
        for lilunzhi_params, lilunzhi in lilunzhi_data:
            for npa_params, npa in nparesult_data:
                # 如果参数完全相同
                if (is_approximately_equal(lilunzhi_params['alpha'], npa_params['alpha']) and
                        is_approximately_equal(lilunzhi_params['p00'], npa_params['p00']) and
                        is_approximately_equal(lilunzhi_params['p01'], npa_params['p01']) and
                        is_approximately_equal(lilunzhi_params['p10'], npa_params['p10']) and
                        is_approximately_equal(lilunzhi_params['p11'], npa_params['p11'])):

                    # 如果lilunzhi的结果不为99999且大于npa的结果
                    if lilunzhi != 99999 and lilunzhi > npa:
                        # 计算结果差值
                        difference = lilunzhi - npa
                        # 写入符合条件的结果到文件
                        output.write(
                            f"alpha={lilunzhi_params['alpha']}, p00={lilunzhi_params['p00']}, p01={lilunzhi_params['p01']}, p10={lilunzhi_params['p10']}, p11={lilunzhi_params['p11']}, lilunzhi={lilunzhi}, npa={npa}, difference={difference}\n")
                        # 调试：输出符合条件的结果
                        print(
                            f"alpha={lilunzhi_params['alpha']}, p00={lilunzhi_params['p00']}, p01={lilunzhi_params['p01']}, p10={lilunzhi_params['p10']}, p11={lilunzhi_params['p11']}, lilunzhi={lilunzhi}, npa={npa}, difference={difference}")


if __name__ == "__main__":
    # 调用比较函数，传入文件路径
    compare_results("lilunBianlip10p111.txt", "nparesultp10p111.txt", "lilunshijip10p111.txt")
# from 10843