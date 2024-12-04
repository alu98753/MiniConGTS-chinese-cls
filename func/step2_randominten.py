import json
import random

# 定義修改intensity的函數
def modify_intensity(triples):
    for triple in triples:
        sentiment = triple["sentiment"]
        if sentiment == "positive":
            # 生成兩個6到10之間的隨機浮點數
            intensity = f"{random.uniform(6, 10):.1f}#{random.uniform(6, 10):.1f}"
        elif sentiment == "negative":
            # 生成兩個1到4之間的隨機浮點數
            intensity = f"{random.uniform(1, 4):.1f}#{random.uniform(1, 4):.1f}"
        elif sentiment == "neutral":
            # 固定生成5.0#5.0
            intensity = "5.0#5.0"
        else:
            intensity = triple["intensity"]  # 保持原始數值
        triple["intensity"] = intensity

# 指定 JSON 檔案的路徑
file_path = "/mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/dev_with_intensity.json"

# 加載 JSON 檔案
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 修改資料
for item in data:
    modify_intensity(item["triples"])

# 將修改後的資料儲存回檔案
output_path = "/mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/dev_with_intensity_modified.json"
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"資料已修改並儲存至 {output_path}")
