import json
import os
from sklearn.model_selection import train_test_split

# 檔案路徑
input_file_path = "/mnt/md0/chen-wei/zi/MiniConGTS-chinese-cls_2/data/D1/res14/NYCU_raw.json"
train_output_path = "/mnt/md0/chen-wei/zi/MiniConGTS-chinese-cls_2/data/D1/res14/NYCU/NYCU_train.json"
test_output_path = "/mnt/md0/chen-wei/zi/MiniConGTS-chinese-cls_2/data/D1/res14/NYCU/NYCU_test.json"

# 讀取 JSON 檔案
with open(input_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 計算每個樣本的三元組數量
data_with_triples = [(item, len(item["triples"])) for item in data]

# 排序資料集，確保數據分布均勻
data_with_triples.sort(key=lambda x: x[1], reverse=True)

# 提取樣本列表
sorted_data = [item[0] for item in data_with_triples]

# 計算分割比例
train_size = int(len(sorted_data) * 0.9)

# 分割數據
train_data = sorted_data[:train_size]
test_data = sorted_data[train_size:]

# 儲存分割後的數據
with open(train_output_path, "w", encoding="utf-8") as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open(test_output_path, "w", encoding="utf-8") as file:
    json.dump(test_data, file, ensure_ascii=False, indent=4)

# 計算各數據集的佔比
total_samples = len(sorted_data)
train_percentage = len(train_data) / total_samples * 100
test_percentage = len(test_data) / total_samples * 100

output_ratios = {
    "Total Samples": total_samples,
    "Train Samples": len(train_data),
    "Train Percentage": f"{train_percentage:.2f}%",
    "Test Samples": len(test_data),
    "Test Percentage": f"{test_percentage:.2f}%"
}

print(output_ratios)
