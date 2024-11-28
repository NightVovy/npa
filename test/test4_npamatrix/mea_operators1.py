import itertools


def generate_operators(layer, extra):
    # 基本的1层算子
    base_operators = ['I', 'A0', 'A1', 'B0', 'B1']
    print(f"Base operators: {base_operators}")  # 调试输出

    # 如果额外层是 "AB" 且层数是1
    if extra == "AB" and layer == 1:
        additional_operators = ['A0B0', 'A0B1', 'A1B0', 'A1B1']
        total_operators = base_operators + additional_operators
        print(f"1-layer + AB extra operators: {total_operators}")  # 调试输出
        return total_operators

    # 如果层数大于1，生成更高层次的组合算子
    if layer >= 2:
        total_operators = base_operators.copy()
        print(f"Start with base operators: {total_operators}")  # 调试输出
        for i in range(1, layer):
            # 合并一层的所有算子与当前的算子
            new_operators = generate_operator_combinations(total_operators, base_operators)
            total_operators = total_operators + new_operators
            print(f"Operators after combining with layer {i + 1}: {total_operators}")  # 调试输出
        return total_operators

    return base_operators


def generate_operator_combinations(operators, base_operators):
    new_combinations = []
    for op1 in operators:
        for op2 in base_operators:
            result = combine_operators(op1, op2)
            if result:
                new_combinations.append(result)
    print(f"New combinations: {new_combinations}")  # 调试输出
    return new_combinations


# def combine_operators(op1, op2):
#     # 检查AxBy形式的算子，如果出现ByAx形式，交换顺序并保证尾号递增
#     if 'A' in op1 and 'B' in op2:
#         # 处理情况：如果 op1 是 A 系列，op2 是 B 系列
#         if op1[0] == 'A' and op2[0] == 'B':
#             # 提取尾号
#             ax_num = int(op1[1:])
#             by_num = int(op2[1:])
#
#             # 如果尾号顺序不正确（即 B 在 A 前面），交换顺序并确保尾号递增
#             if ax_num > by_num:
#                 # 如果尾号递减，交换顺序
#                 return op2 + op1  # 将 B 放在 A 前面，形成 ByAx
#             else:
#                 return op1 + op2  # A 在前，B 在后
#     return None
def combine_operators(op1, op2):
    """
    检查相邻算子的乘积是否为 I，如果是则消去算子。
    """
    # 处理 A 和 B 系列算子的组合
    if 'A' in op1 and 'B' in op2:
        # 如果 op1 是 A 系列，op2 是 B 系列，直接返回组合
        return op1 + op2
    # 检查是否为相同算子（如 A0A0 或 B1B1），若是则返回 'I'
    elif op1 == op2:
        return 'I'

    return None


def check_hermitian(operators):
    # 确保算子是厄米的
    # 1. 检查算子是否是其自身的共轭转置
    # 2. 确保算子自乘结果是单位算子I
    for op in operators:
        if op[0] == 'A':  # Alice算子需要满足A* * A = I
            if op[0] != op[::-1]:  # 简单地示意检查共轭转置
                print(f"Error: {op} is not Hermitian!")
                return False
        elif op[0] == 'B':  # Bob算子需要满足B* * B = I
            if op[0] != op[::-1]:  # 简单地示意检查共轭转置
                print(f"Error: {op} is not Hermitian!")
                return False
    return True


def sort_operators(operators):
    """
    对测量算子进行排序，确保按照字母部分和尾号部分排序，处理相邻算子的消去逻辑。
    """

    def sort_key(op):
        # 如果算子是 "I"，直接返回一个特殊值，确保它排在最前面
        if op == 'I':
            return ('I', -1)

        # 对于其他算子，提取尾号数字部分
        prefix = ''.join([ch for ch in op if not ch.isdigit()])
        suffix = ''.join([ch for ch in op if ch.isdigit()])

        # 如果没有尾号数字（如 'A' 或 'B'），我们给它一个默认值 -1，放在前面
        if not suffix:
            return (prefix, -1)

        # 否则，返回字母部分和尾号数字部分
        return (prefix, int(suffix))

    # 对算子进行排序
    sorted_operators = sorted(operators, key=sort_key)

    # 消去相邻的相同算子
    final_operators = []
    for i in range(len(sorted_operators)):
        if i > 0 and sorted_operators[i] == sorted_operators[i - 1]:
            # 如果当前算子与前一个算子相同（如 A0A0），将其消去并替换为 "I"
            if sorted_operators[i] != 'I':
                final_operators[-1] = 'I'  # 把前一个算子变为 I
        else:
            final_operators.append(sorted_operators[i])

    # 如果最终算子中有非 "I" 的算子，移除 "I"
    if any(op != 'I' for op in final_operators):
        final_operators = [op for op in final_operators if op != 'I']

    return final_operators


def set_previous_function(func_name):
    global previous_function
    previous_function = func_name


def get_previous_function():
    global previous_function
    return previous_function


def main():
    # 输入参数
    layer = 2  # 可以根据需求修改
    extra = "AB"  # 可以根据需求修改

    # 生成测量算子
    operators = generate_operators(layer, extra)
    set_previous_function("operator_generator")

    # 检查厄米算子条件
    if check_hermitian(operators):
        print("All operators are Hermitian and satisfy the conditions.")
    else:
        print("Some operators do not satisfy Hermitian conditions.")

    # 对算子进行排序
    sorted_operators = sort_operators(operators)
    set_previous_function("operator_sorting")

    # 输出排序后的算子
    print("Sorted Operators:")
    for op in sorted_operators:
        print(op)

    # 获取上一个执行的功能
    last_function = get_previous_function()
    print(f"The last executed function was: {last_function}")


if __name__ == "__main__":
    previous_function = None
    main()
