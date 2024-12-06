import json
import matplotlib.pyplot as plt
from collections import Counter

# 讀取 JSON 文件
file_path = r"E:\NYCU-Project\Class\NLP\MiniConGTS-chinese-cls\data\D1\res14\NYCU_NLP_113A_TrainingSet.json"

# 加載 JSON 文件
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化分類統計
intensity_v = []
intensity_a = []

# 分類數據
for item in data:
    if "Intensity" in item and item["Intensity"]:
        for intensity in item["Intensity"]:
            # 提取範圍的起始值和結束值
            v, a = map(float, intensity.split('#'))
            v = round(v)
            a = round(a)
            # 計算範圍的中心值，取整
            intensity_v.append(v)
            intensity_a.append(a)

# 計算 v 和 a 的比例
def calculate_ratios(intensity_list):
    total_count = len(intensity_list)
    counter = Counter(intensity_list)
    return {key: value / total_count for key, value in counter.items()}

ratios_v = calculate_ratios(intensity_v)
ratios_a = calculate_ratios(intensity_a)

# 打印結果
print("Intensity 分布比例 (v):")
for intensity, ratio in sorted(ratios_v.items()):
    print(f"Intensity: {intensity}, 比例: {ratio:.2%}")

print("\nIntensity 分布比例 (a):")
for intensity, ratio in sorted(ratios_a.items()):
    print(f"Intensity: {intensity}, 比例: {ratio:.2%}")

# 繪製分布圖
plt.bar(ratios_v.keys(), ratios_v.values(), edgecolor='black', alpha=0.7, label='v')
plt.bar(ratios_a.keys(), ratios_a.values(), edgecolor='black', alpha=0.5, label='a')
plt.title("Intensity Distribution (Rounded)")
plt.xlabel("Intensity (Rounded)")
plt.ylabel("Proportion")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(0, 11))  # 顯示 0-10 的刻度
plt.legend()
plt.show()
