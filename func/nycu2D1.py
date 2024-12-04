import os
import json

# 定義讀取檔案路徑和輸出檔案路徑
input_file_path = "/mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/NYCU_NLP_113A_TrainingSet.json"
output_file_path = "/mnt/md0/chen-wei/zi/MiniConGTS_copy/data/D1/res14/NYCU_raw.json"


# 確認輸入檔案是否存在
if os.path.exists(input_file_path):
    # 讀取 JSON 檔案
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    

    # def validate_from_to(sentence, from_to_list):
    #     for ft in from_to_list:
    #         start, end = map(int, ft.split("#"))
    #         if start < 0 or end > len(sentence):
    #             print(f"Invalid range: {start}#{end} for sentence length {len(sentence)}")
    #             return False
    #     return True

    # 定義轉換函數
    def convert_to_d1_format(data):
        import spacy
        nlp = spacy.load("zh_core_web_sm")

        def generate_bio_tags_with_text(sentence, from_to, tag_type="B"):
            tags = [f"{char}\\O" for char in sentence]
            for ft in from_to:
                start, end = map(int, ft.split("#"))
                if start < 0 or end > len(sentence):
                    print(f"Warning: Invalid range {start}#{end} for sentence: {sentence}")
                    continue
                tags[start-1] = f"{sentence[start-1]}\\{tag_type}"  # 標記起點為 B
                for i in range(start , end):
                    tags[i] = f"{sentence[i]}\\I"
            return "\\".join(tags)


        def determine_sentiment(intensity):
            values = list(map(float, intensity.split("#")))
            avg_intensity = sum(values) / len(values)
            if avg_intensity < 5:
                return "negative"
            elif avg_intensity == 5:
                return "neutral"
            else:
                return "positive"

        d1_data = []
        for item in data:
            sentence = item["Sentence"]
            doc = nlp(sentence)

            triples = []
            num_aspects = len(item["Aspect"])
            num_opinions = len(item["Opinion"])
            num_intensities = len(item["Intensity"])

            # 確保資料長度一致
            min_length = min(num_aspects, num_opinions, num_intensities)

            for i in range(min_length):
                target_from_to = [item["AspectFromTo"][i]]
                opinion_from_to = [item["OpinionFromTo"][i]]

                # if not validate_from_to(sentence, target_from_to) or not validate_from_to(sentence, opinion_from_to):
                #     print(f"Skipping triple with ID: {item['ID']}-{i} due to invalid ranges")
                #     continue

                triples.append({
                    "uid": f'{item["ID"]}-{i}',
                    "target_tags": generate_bio_tags_with_text(sentence, target_from_to, tag_type="B"),
                    "opinion_tags": generate_bio_tags_with_text(sentence, opinion_from_to, tag_type="B"),
                    "sentiment": determine_sentiment(item["Intensity"][i]),
                    "intensity": item["Intensity"][i]
                })


            d1_item = {
                "id": item["ID"],
                "sentence": sentence,
                "postag": [token.pos_ for token in doc],
                "head": [token.head.i + 1 for token in doc],
                "deprel": [token.dep_ for token in doc],
                "triples": triples
            }
            d1_data.append(d1_item)
        return d1_data


    # 轉換資料
    converted_data = convert_to_d1_format(data)

    # 儲存轉換後的資料到新檔案
    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(converted_data, file, ensure_ascii=False, indent=4)

    print(f"Data converted and saved to {output_file_path}")
else:
    print(f"Input file {input_file_path} does not exist.")
