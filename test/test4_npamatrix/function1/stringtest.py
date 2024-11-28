def generate_string_sets(layer, S1):
    # 初始集合 S1
    if layer == 1:
        return S1

    # 获取上一层的集合 Sn
    Sn = generate_string_sets(layer - 1, S1)

    # 当前层的集合 S(n+1)
    Sn_plus_1 = set(Sn)  # 先包含上一层的所有元素

    # 对每个 Sn 中的元素与 S1 中的元素进行右侧组合
    for s in Sn:
        for s1 in S1:
            Sn_plus_1.add(s + s1)

    return Sn_plus_1


# 初始化 S1
S1 = {"I", "A0", "A1", "B0", "B1"}

# 获取集合 S2 和 S3
S2 = generate_string_sets(2, S1)
S3 = generate_string_sets(3, S1)

print("S2:", S2)
print("S3:", S3)
