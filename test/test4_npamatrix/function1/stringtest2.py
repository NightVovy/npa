from collections import Counter


# 处理字符串的函数
def process_strings(Sn_plus_1):
    # 第一个循环: 处理字符串中的 'I'
    processed_set = set()
    for s in Sn_plus_1:
        # 如果字符串中只包含 I，保留
        if all(c == 'I' for c in s):
            processed_set.add('I')
        else:
            # 否则删除所有 I
            processed_set.add(s.replace('I', ''))

    # 第二个循环: 筛选偶数个的元素
    final_set = set()
    segments = ['I', 'A0', 'A1', 'B0', 'B1']

    for s in processed_set:
        # 统计每个小段的出现次数
        counter = {segment: 0 for segment in segments}  # 初始化计数器
        for segment in segments:
            counter[segment] = s.count(segment)  # 统计每个小段在字符串中出现的次数

        # 构建新的字符串
        new_str = []
        for segment in segments:
            # 如果该段出现次数为偶数，则删除该段
            count = counter[segment]
            if count % 2 != 0:  # 保留出现奇数次的段
                new_str.append(segment * (count % 2))  # 保留奇数个

        # 如果所有段都被删除，最终字符串设为 'I'
        final_str = ''.join(new_str)
        if not final_str:  # 如果没有任何字符，则设为 'I'
            final_str = 'I'

        final_set.add(final_str)

    # 第三个循环: 根据字符串的长度从小到大排序
    def custom_sort_key(s):
        # 计算每个小段在字符串中是否出现，按照 A0 > A1 > B0 > B1 的顺序
        order = {'A0': 0, 'A1': 1, 'B0': 2, 'B1': 3}
        key = []

        for segment in ['A0', 'A1', 'B0', 'B1']:
            # 如果该段出现在字符串中，则加上一个优先级，若没有则加一个较大值（确保不出现的段排后）
            key.append(order[segment] if segment in s else len(order))

        return (len(s), key)  # 先按长度排序，长度相同则按小段顺序排序

    # 使用自定义排序键排序字符串集合
    sorted_set = sorted(final_set, key=custom_sort_key)

    return sorted_set


# 示例输入
S1 = {"I", "A0", "A1", "B0", "B1"}


def generate_string_sets(layer, S1):
    if layer == 1:
        return S1

    Sn = generate_string_sets(layer - 1, S1)
    Sn_plus_1 = set(Sn)  # 先包含上一层的所有元素

    # 生成当前层的元素
    for s in Sn:
        for s1 in S1:
            Sn_plus_1.add(s + s1)

    # 整理字符串集合
    return process_strings(Sn_plus_1)


# 获取集合 S2 和 S3，并整理
S2 = generate_string_sets(2, S1)
S3 = generate_string_sets(3, S1)

print("S2:", S2)
print("S3:", S3)
