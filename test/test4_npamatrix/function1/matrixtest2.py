from collections import defaultdict
from stringtest2 import generate_string_sets  # 从 stringtest2.py 中导入 generate_string_sets 函数


# 处理字符串的函数
def process_string(s):
    # 第一个循环: 处理字符串中的 'I'
    if all(c == 'I' for c in s):
        return 'I'
    else:
        s = s.replace('I', '')  # 删除所有的 'I'

    # 第二个循环: 筛选偶数个的元素
    segments = ['I', 'A0', 'A1', 'B0', 'B1']
    counter = {segment: 0 for segment in segments}

    for segment in segments:
        counter[segment] = s.count(segment)

    new_str = []
    for segment in segments:
        count = counter[segment]
        if count % 2 != 0:
            new_str.append(segment * (count % 2))  # 保留出现奇数次的段

    # 如果所有段都被删除，最终字符串设为 'I'
    final_str = ''.join(new_str)
    if not final_str:
        final_str = 'I'

    return final_str


# 创建上三角矩阵
def create_upper_triangle_matrix(S):
    # 将字符串集合转换为列表，方便索引
    S_list = list(S)
    n = len(S_list)

    # 初始化一个 n x n 的矩阵，值为 None
    matrix = [[None] * n for _ in range(n)]

    # 遍历每个字符串集合，按照规则拼接并填充矩阵
    for i in range(n):
        for j in range(i, n):  # 只处理上三角部分，包括对角线
            if i == j:
                matrix[i][j] = S_list[i] + S_list[i]  # 自己和自己拼接
            else:
                matrix[i][j] = S_list[j] + S_list[i]  # 拼接第二个字符串在前，第一个字符串在后

            # 整理拼接后的新字符串
            matrix[i][j] = process_string(matrix[i][j])

    return matrix


# 排序规则：首先按照字符串长度排序，其次按 A0 > A1 > B0 > B1
def sort_key(s):
    # 将字符串分解成段
    segments = ['A0', 'A1', 'B0', 'B1']
    segment_count = {segment: s.count(segment) for segment in segments}

    # 按照长度排序，接着按 A0, A1, B0, B1 出现的次数顺序排序
    return (len(s), segment_count['A0'], segment_count['A1'], segment_count['B0'], segment_count['B1'])


# 记录矩阵中的元素、出现次数和位置
def record_matrix_elements(matrix):
    # 使用 defaultdict 来记录每个元素出现的次数和位置
    element_info = defaultdict(lambda: {'count': 0, 'positions': []})

    n = len(matrix)

    for i in range(n):
        for j in range(i, n):  # 只处理上三角部分
            element = matrix[i][j]
            if element != 'I':  # 如果是 'I'，跳过
                element_info[element]['count'] += 1
                element_info[element]['positions'].append([i, j])

    # 对元素按要求排序
    sorted_elements = sorted(element_info.items(), key=lambda x: sort_key(x[0]))

    # 输出排序后的元素、出现次数和位置
    for element, info in sorted_elements:
        print(f"Element: {element}")
        print(f"  Count: {info['count']}")
        print(f"  Positions: {info['positions']}")


# 打印矩阵，确保每行元素对齐
def print_matrix(matrix):
    # 找到矩阵中最大长度的元素
    max_length = max(len(str(item)) for row in matrix for item in row if item is not None)

    for row in matrix:
        # 打印每行时，确保每个元素的宽度一致
        print("  ".join(f"{str(item):<{max_length}}" if item is not None else " " * max_length for item in row))


# 示例输入
S1 = {"I", "A0", "A1", "B0", "B1"}

# 生成 S2 或 S3 的字符串集合（直接调用 generate_string_sets）
layer = 2  # 可根据需求修改层数
S = generate_string_sets(layer, S1)  # 使用从 stringtest2.py 导入的函数

# 输出生成的上三角矩阵
print(f"S{layer} Upper Triangle Matrix:")
matrix = create_upper_triangle_matrix(S)
print_matrix(matrix)

# 记录矩阵中的元素、出现次数和位置，并输出
record_matrix_elements(matrix)
