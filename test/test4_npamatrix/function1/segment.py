from collections import Counter

def count_segments(s, segments):
    # 统计各个小段的数量
    counter = {segment: 0 for segment in segments}  # 初始化计数器
    for segment in segments:
        counter[segment] = s.count(segment)  # 统计每个小段在字符串中出现的次数
    return counter

# 示例字符串
s = "A0A1B0IA0A0B1A0B1"
segments = ['I', 'A0', 'A1', 'B0', 'B1']

# 统计每个小段出现的次数
counter = count_segments(s, segments)

print(counter)
