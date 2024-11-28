def generate_measurement_operators(layer, extra=""):
    """
    根据层数 `layer` 和 `extra` 生成测量算子集合
    """

    def optimize_operators(operators):
        """
        优化算子集合：
        1. 排序算子。
        2. 检查相同算子相乘的情况（如 A0A0, B1B1），优化为 I。
        3. 如果优化后的集合中只剩一个 I，保留一个 I；否则删除所有的 I。
        """

        # 1. 排序算子
        def sort_key(op):
            # 对于 "I"，我们让它排在最前面
            if op == 'I':
                return ('I', 0)
            # 否则，我们根据字母和数字部分进行排序
            prefix = op[0]  # 字母部分
            number = int(op[1:]) if op[1:].isdigit() else float('inf')  # 数字部分
            return (prefix, number)

        sorted_operators = sorted(operators, key=sort_key)

        # 2. 检查相同算子相乘的情况并优化为 I
        final_operators = []
        for i in range(len(sorted_operators)):
            if i > 0 and sorted_operators[i] == sorted_operators[i - 1]:
                # 如果当前算子与前一个算子相同（如 A0A0 或 B1B1），将其消去并替换为 "I"
                if sorted_operators[i] != 'I':
                    final_operators[-1] = 'I'  # 把前一个算子变为 I
            else:
                final_operators.append(sorted_operators[i])

        # 3. 如果优化后的集合中只有一个 I，保留一个 I；否则删除所有 I
        if 'I' in final_operators:
            # 如果有 I，并且优化后的集合只有 I，保留一个 I
            if len(final_operators) == 1:
                final_operators = ['I']
            else:
                final_operators = [op for op in final_operators if op != 'I']

        return final_operators

    def combine_operators(op1, op2):
        """
        处理算子乘积，检查是否有相邻的相同算子（如 A0A0），如果是则返回 "I"
        """
        # 处理 A 和 B 系列算子的组合
        if 'A' in op1 and 'B' in op2:
            return op1 + op2  # 例如 "A0B0"
        # 检查是否为相同算子（如 A0A0 或 B1B1），若是则返回 'I'
        elif op1 == op2:
            return 'I'
        return None

    # 构造基本的测量算子
    base_operators = ['I', 'A0', 'A1', 'B0', 'B1']

    # 处理 layer 为 1 且 extra 为 "AB" 的特殊情况
    if layer == 1 and extra == "AB":
        operators = base_operators + ['A0B0', 'A0B1', 'A1B0', 'A1B1']
    else:
        # 对于 layer >= 1，按照上一层和第一层的乘积生成新的算子
        operators = base_operators
        for l in range(2, layer + 1):
            new_operators = []
            for op1 in operators:
                for op2 in base_operators:
                    combined_op = combine_operators(op1, op2)
                    if combined_op:
                        new_operators.append(combined_op)
            operators += new_operators
        # 优化算子集合
        operators = optimize_operators(operators)

    # 确保最终算子集合中只有一个 "I"
    if 'I' in operators:
        operators = ['I'] + [op for op in operators if op != 'I']

    return operators


# 调用函数生成测量算子集合
layer = 2  # 举例输入层数
operators = generate_measurement_operators(layer)
print(operators)

layer = 3  # 举例输入层数
operators = generate_measurement_operators(layer)
print(operators)
