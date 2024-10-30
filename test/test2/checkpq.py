# 别用了p用没有
def check_relationship(p00, p01, p10, p11):
    total = p00 + p01 + p10 + p11

    if total != 1:
        return False, "p00, p01, p10, p11 的总和不为 1"

    a = p00 + p01
    b = p00 + p10

    if 0 <= a <= 1 and 0 <= b <= 1:
        # 验证关系
        p00_check = a * b
        p01_check = a * (1 - b)
        p10_check = (1 - a) * b
        p11_check = (1 - a) * (1 - b)

        return (p00_check == p00 and
                p01_check == p01 and
                p10_check == p10 and
                p11_check == p11), (a, b)
    else:
        return False, "a 或 b 超出范围"


# 示例输入
p00, p01, p10, p11 = 0.15, 0.15, 0.15, 0.15
result, values = check_relationship(p00, p01, p10, p11)
print(result, values)
