from transformers import BertTokenizer
import jieba

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# print(tokenizer.tokenize("測試一下中文分詞，漢堡真好吃"))
sentence = "這是一個中文分詞測試"
words = jieba.lcut(sentence)
print(words)
print(" ".join(words))  # 用空格分隔詞
# Output: '這 是 一個 中文 分詞 測試'
tokens = tokenizer.tokenize(" ".join(words))
print(tokens)