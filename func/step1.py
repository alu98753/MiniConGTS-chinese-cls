import json

# 讀取現有的 train.json
# train # /mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14_t/train.json
# dev   # /mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/test.json
# test  # /mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/dev.json
with open('/mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 自訂 sentiment -> intensity 的映射（僅供示例）
sentiment_to_intensity = {
    "positive": (7.5, 7.0),
    "neutral": (5.0, 5.0),
    "negative": (3.0, 3.5)
}

# 新增 intensity 欄位並移除 sentiment
for item in data:
    for triple in item.get("triples", []):
        sentiment = triple.get("sentiment", "neutral")  # 預設為 neutral
        intensity = sentiment_to_intensity.get(sentiment, (5.0, 5.0))
        triple["intensity"] = f"{intensity[0]}#{intensity[1]}"
        # del triple["sentiment"]  # 刪除 sentiment 欄位

for item in data :
    print(item)

# 將結果保存為新的 JSON
with open('/mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/dev_with_intensity.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("資料處理完成，結果已儲存為 train_with_intensity.json")
