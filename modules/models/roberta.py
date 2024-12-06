import torch
from transformers import RobertaModel

import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.bert = RobertaModel.from_pretrained(args.model_name_or_path)
        self.norm0 = torch.nn.LayerNorm(args.bert_feature_dim)
        self.drop_feature = torch.nn.Dropout(0.1)

        self.linear1 = torch.nn.Linear(args.bert_feature_dim*2, self.args.max_sequence_len)
        self.norm1 = torch.nn.LayerNorm(self.args.max_sequence_len)
        self.cls_linear = torch.nn.Linear(self.args.max_sequence_len, args.class_num)  # sentiment分類頭
        self.cls_linear1 = torch.nn.Linear(self.args.max_sequence_len, 2)  # opinion分類頭

        # 用於預測 Valence 和 Arousal 的分類頭，每個輸出 11 個類別 (0-10)
        # 用於預測 Valence 和 Arousal 的分類頭，輸出五個類別 (0-4)
        self.cls_linear_valence = torch.nn.Linear(self.args.max_sequence_len, 6)  # Valence 分類頭
        self.cls_linear_arousal = torch.nn.Linear(self.args.max_sequence_len, 6)  # Arousal 分類頭


        self.gelu = torch.nn.GELU()

    def forward(self, tokens, masks):
        # print("tokens shape:", tokens.shape)
        # print("masks shape:", masks.shape)
        bert_feature, _ = self.bert(tokens, masks, return_dict=False)
        # print("調試點1 BERT Output Shape:", bert_feature.shape)  # 調試點1
        # print("調試點1 BERT Output Values:", bert_feature[:2, :2, :5])  # 打印部分值避免過多輸出

        
        bert_feature = self.norm0(bert_feature)
        # bert_feature = self.drop_feature(bert_feature)  # 对 bert 后的特征表示做 dropout
        bert_feature = bert_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        bert_feature_T = bert_feature.transpose(1, 2)
        
        features = torch.cat([bert_feature, bert_feature_T], dim=3)
        
        
        sim_matrix = torch.nn.functional.cosine_similarity(bert_feature, bert_feature_T, dim=3)
        # print(sim_matrix.shape)
        sim_matrix = sim_matrix * masks

        # print(sim_matrix.shape, masks.shape)
        
        hidden = self.linear1(features)
        # hidden = self.drop_feature(hidden)
        hidden = self.norm1(hidden)
        hidden = self.gelu(hidden)

        logits = self.cls_linear(hidden) # sentiment分類輸出
        logits1 = self.cls_linear1(hidden) # opinion分類輸出
        # intensity回歸輸出
        
        # Valence 的分類頭
        logits_valence = self.cls_linear_valence(hidden)
        print(f"logits_valence{logits_valence.shape}")

        # Arousal 的分類頭
        logits_arousal = self.cls_linear_arousal(hidden)

        # print("調試點2 Intensity Scores Shape:", intensity_scores.shape)  # 調試點2
        # print("調試點2 Intensity Scores Values:", intensity_scores[:5])  # 打印部分值
        
        masks0 = masks.unsqueeze(3).expand([-1, -1, -1, self.args.class_num])#.shape
        masks1 = masks.unsqueeze(3).expand([-1, -1, -1, 2])#.shape
        masks_valence = masks.unsqueeze(3).expand([-1, -1, -1, 6])
        masks_arousal = masks.unsqueeze(3).expand([-1, -1, -1, 6])

        # 使用 masks 過濾無效位置的 logits
        logits = masks0 * logits
        logits1 = masks1 * logits1
        logits_valence = masks_valence * logits_valence
        logits_arousal = masks_arousal * logits_arousal
        
        return logits, logits1, sim_matrix, logits_valence, logits_arousal
        # return logits, logits1, sim_matrix
    
