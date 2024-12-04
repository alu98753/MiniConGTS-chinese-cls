import numpy as np
import torch
import torch.nn.functional as F

class Metric():
    '''评价指标 precision recall f1'''
    # def __init__(self, args, stop_words, tokenized, ids, predictions, goldens, sen_lengths, tokens_ranges, ignore_index=-1, logging=print):
        # metric = Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        # print([i.sum() for i in predictions], [i.sum() for i in goldens])
    def __init__(self, args, stop_words, tokenized, ids, predictions, goldens, sen_lengths, 
                 tokens_ranges, intensities, predicted_intensities, ignore_index=-1, logging=print):    
        _g = np.array(goldens)
        _g[_g==-1] = 0
        
        print('sum_pred: ', np.array(predictions).sum(), ' sum_gt: ', _g.sum())
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        # self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = ignore_index
        self.data_num = len(self.predictions)
        self.ids = ids
        self.stop_words = stop_words
        self.tokenized = tokenized
        self.logging = logging
        
        self.intensities = intensities  # 真實值
        self.predicted_intensities = predicted_intensities  # 預測值

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_triplet_golden(self, tag):
        triplets = []
        # print(f"調試 tag 矩陣：\n{tag}")

        for row in range(1, tag.shape[0]-1):
            for col in range(1, tag.shape[1]-1):
                if row==col:
                    pass
                elif tag[row][col] in self.args.sentiment2id.values():
                    # print(f"匹配成功：row={row}, col={col}, value={tag[row][col]}")

                    sentiment = int(tag[row][col])
                    al, pl = row, col
                    ar = al
                    pr = pl
                    while tag[ar+1][pr] == 1:
                        ar += 1
                    while tag[ar][pr+1] == 1:
                        pr += 1

                    triplets.append([al, ar, pl, pr, sentiment])
                    # print(f"匹配 triplet: {[al, ar, pl, pr, sentiment]}")

                # print(triplets)
                # [[1, 3, 6, 6, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5], [16, 16, 15, 15, 5]]
                
        return triplets
    # 調試：對應 golden_tuple 與 find_triplet 結果
    def compare_triplets(self, golden_tuple, tokenized, tokens_ranges):
        for i in range(len(golden_tuple)):
            print(f"golden_tuple:{golden_tuple[i]}")
            al, ar, pl, pr, sentiment = golden_tuple[i]
            print(f"golden_tuple[i]:{golden_tuple[i]}")
            aspect_range = tokens_ranges[i][al:ar ]
            opinion_range = tokens_ranges[i][pl:pr ]
            print(f"對應檢查:")
            print(f"Aspect words: {tokenized[al:ar]}, Expected range: {aspect_range}")
            print(f"Opinion words: {tokenized[pl:pr]}, Expected range: {opinion_range}")

    # def find_triplet_golden(self, tag):
    #     triplets = []
    #     for row in range(tag.shape[0]):
    #         for col in range(tag.shape[1]):
    #             if tag[row][col] in self.args.sentiment2id.values():
    #                 # print(f"匹配成功：row={row}, col={col}, value={tag[row][col]}")

    #                 sentiment = tag[row][col]
    #                 triplets.append([row, row, col, col, sentiment])
    #     return triplets

    
    def find_triplet(self, tag, ws, tokenized):
        triplets = []
        # print(f"調試：tag 矩陣\n{tag}")
        # print(f"調試：tokens_ranges={ws}")
        # print(f"調試：tokenized sentence={tokenized}")

        for row in range(1, tag.shape[0]-1):
            for col in range(1, tag.shape[1]-1):
                if row==col:
                    pass
                elif tag[row][col] in self.args.sentiment2id.values():
                    sentiment = int(tag[row][col])
                    al, pl = row, col
                    ar = al
                    pr = pl
                    while tag[ar+1][pr] == 1:
                        ar += 1
                    while tag[ar][pr+1] == 1:
                        pr += 1
                    
                    '''filting the illegal preds'''
                    #目標： 確保 (al, ar) 和 (pl, pr) 對應的索引落在 ws (word spans) 的範圍內
                    condition1 = al in np.array(ws)[:, 0] and ar in np.array(ws)[:, 1] and pl in np.array(ws)[:, 0] and pr in np.array(ws)[:, 1]
                    
                    #目標： 確保 aspect 和 opinion 的範圍沒有交疊。
                    condition2 = True
                    for ii in range(al, ar+1):
                        for jj in range(pl, pr+1):
                            if ii == jj:
                                condition2 = False
                                
                    #目標： 確保 aspect 的 tokens 不包含停用詞（stop_words）
                    condition3 = True
                    # for tk in tokenized[al: ar+1]:
                    #     # print(tk)
                    #     if tk in self.stop_words:
                    #         condition3 = False
                    #         break
                    
                    #目標： 確保 opinion 的 tokens 不包含停用詞。
                    condition4 = True
                    # for tk in tokenized[pl: pr+1]:
                    #     # print(tk)
                    #     if tk in self.stop_words:
                    #         condition4 = False
                    #         break

                    conditions = condition1 and condition2 and condition3 and condition4                        
                    # conditions = condition1 and condition2

                    if conditions:
                        # print(f"Triplet found: al={al}, ar={ar}, pl={pl}, pr={pr}, sentiment={sentiment}")
                        # print(f"Aspect range: {tokenized[al:ar+1]}, Opinion range: {tokenized[pl:pr+1]}")

                        triplets.append([al, ar, pl, pr, sentiment])

                # print(triplets)
                # [[1, 3, 6, 6, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5]]
                # [[1, 3, 6, 6, 5], [9, 11, 13, 13, 5], [16, 16, 15, 15, 5]]
        if not triplets:
            return triplets

        # self.compare_triplets(triplets, tokenized, ws)        
        return triplets
    
    # def get_sets(self):
    #     assert len(self.predictions) == len(self.goldens)
    #     golden_set = set()
    #     predicted_set = set()
    #     for i in range(self.data_num):
    #         # golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
    #         # golden_opinion_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
    #         id = self.ids[i]
    #         golden_tuples = self.find_triplet_golden(np.array(self.goldens[i]))
    #         # golden_tuples: triplets.append([al, ar, pl, pr, sentiment])
    #         for golden_tuple in golden_tuples:
    #             golden_set.add(id + '-' + '-'.join(map(str, golden_tuple)))  # 从前到后把得到的三元组纳入总集合
    #             # golden_set: ('0-{al}-{ar}-{pl}-{pr}-{sentiment}', '1-{al}-{ar}-{pl}-{pr}-{sentiment}', '2-{al}-{ar}-{pl}-{pr}-{sentiment}')

    #         # predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
    #         # predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
    #         # if self.args.task == 'pair':
    #         #     predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
    #         # elif self.args.task == 'triplet':
                
    #         tag = np.array(self.predictions[i])

    #         tag[0][:] = -1
    #         tag[-1][:] = -1
    #         tag[:, 0] = -1
    #         tag[:, -1] = -1

    #         predicted_triplets = self.find_triplet(tag, self.tokens_ranges[i], self.tokenized[i])  # , predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i]
    #         for pair in predicted_triplets:
    #             predicted_set.add(id + '-' + '-'.join(map(str, pair)))
    #     return predicted_set, golden_set

    #改動起點
    def get_sets(self):
        assert len(self.predictions) == len(self.goldens)
        p_golden_set = []
        p_predicted_set = []
        golden_set = set()
        predicted_set = set()
        print(f"self.data_num: {self.data_num}, len(self.intensities): {len(self.intensities)}")

        for i in range(self.data_num):
            id = self.ids[i]
            tokenized_sentence = self.tokenized[i]
            golden_tuples = self.find_triplet_golden(np.array(self.goldens[i]))
            
            # print(f"tokens_ranges: {self.tokens_ranges[i]}")
            # print(f"tokenized sentence: {self.tokenized[i]}")
            for golden_tuple in golden_tuples:
                # print(golden_tuple)
                # print(f"self.intensities[{i}]: {self.intensities[i]}, type: {type(self.intensities[i])}")
                # # 对 intensity 进行乘以 10 后取整
                
                # print(f"調試 golden_tuple 的形狀: {torch.tensor(self.intensities[i][0]).shape}")
                # print(f"調試 golden_tuple : {self.intensities[i][0]}")

                rounded_intensity = list(map(int, torch.round(torch.tensor(self.intensities[i][0]) ).tolist()))

                golden_set.add(id + '-' + '-'.join(map(str, golden_tuple)) + '-' + '-'.join(map(str, rounded_intensity)))
                # print("調試點golden_tuples_triplets:",id + '-' + '-'.join(map(str, golden_tuple)) + '-' + '-'.join(map(str, rounded_intensity)))
            # print(f"調試點golden_tuples_triplets:{golden_tuples}")

            # Process golden triplets
            for idx, golden_tuple in enumerate(golden_tuples):
                # Ensure idx is within the range of intensities
                if idx < len(self.intensities[i]):
                    intensity_values = self.intensities[i][idx]  # Already a list
                else:
                    intensity_values = [0.0, 0.0]  # Default value if no intensity
                p_golden_set.append({
                    'id': id,
                    'aspect_indices': (golden_tuple[0], golden_tuple[1]),
                    'opinion_indices': (golden_tuple[2], golden_tuple[3]),
                    'sentiment': golden_tuple[4],
                    'intensity': intensity_values
                })
            
            # Process predicted triplets

            # 獲取 predicted triplets，並新增 Intensity
            tag = np.array(self.predictions[i])
            tag[0][:] = -1
            tag[-1][:] = -1
            tag[:, 0] = -1
            tag[:, -1] = -1
            predicted_triplets = self.find_triplet(tag, self.tokens_ranges[i], self.tokenized[i])

            for pair in predicted_triplets:
                
                # print(f"調試 predicted_triplet 的形狀: {self.intensities[i][0]}")

                # print(f"調試 predicted_triplets 的形狀: {torch.tensor(self.intensities[i][0]).shape}")

                # print(f"調試 predicted intensity: {self.predicted_intensities[i][0]}")

                intensity_scores = list(map(int, torch.round(torch.tensor(self.intensities[i][0]) ).tolist()))
                # print(f"調試 predicted intensity: {intensity_scores[0]}")
                # print(f"調試 predicted intensity: {intensity_scores[0]}")

                predicted_set.add(id + '-' + '-'.join(map(str, pair)) + '-' + '-'.join(map(str, intensity_scores)))


            # print(f"調試點predicted_triplets:{predicted_triplets}")
            
            for idx, predicted_tuple in enumerate(predicted_triplets):
                if idx < len(self.predicted_intensities[i]):
                    intensity_scores = self.predicted_intensities[i][idx]  # Already a list
                else:
                    intensity_scores = [0.0, 0.0]  # Default value if no intensity
                p_predicted_set.append({
                    'id': id,
                    'aspect_indices': (predicted_tuple[0], predicted_tuple[1]),
                    'opinion_indices': (predicted_tuple[2], predicted_tuple[3]),
                    'sentiment': predicted_tuple[4],
                    'intensity': intensity_scores
                })
        
        return p_predicted_set, p_golden_set ,predicted_set, golden_set 


    def extract_and_print_triplets(self, predicted_set):
        sentiment_map = {2: 'negative', 3: 'neutral', 4: 'positive'}
        
        for triplet_info in predicted_set:
            id = triplet_info['id']
            aspect_indices = triplet_info['aspect_indices']
            opinion_indices = triplet_info['opinion_indices']
            sentiment = triplet_info['sentiment']
            intensity = triplet_info['intensity']
            
            # Get the tokenized sentence
            tokenized_sentence = self.tokenized[self.ids.index(id)]
            # Extract aspect and opinion words
            aspect_words = tokenized_sentence[aspect_indices[0]:aspect_indices[1]+1]
            opinion_words = tokenized_sentence[opinion_indices[0]:opinion_indices[1]+1]
            
            # Convert intensity values to strings
            intensity_str = '#'.join([f"{val:.2f}" for val in intensity])
            
            # Format and print the triplet
            aspect = ''.join(aspect_words)
            opinion = ''.join(opinion_words)
            sentiment_label = sentiment_map.get(sentiment, 'unknown')
            
            # print(f"id:{id}, tokenized_sentence{tokenized_sentence}")
            # print(f"aspect_indices{aspect_indices}")
            # print(f"opinion_indices{opinion_indices}")
            # print(f"sentiment{sentiment}")
            # print(f"intensity{intensity}")
            
            # print(f"{id}: ({aspect}, {opinion}, {intensity_str})")


    def score_triplets(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 6]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 6]) for i in golden_set])
       
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Triplet\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1

    def score_pairs(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 5]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 5]) for i in golden_set])
        
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Pair\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1
    
    def score_aspect(self, predicted_set, golden_set):
        predicted_set = set(['-'.join(i.split('-')[0: 3]) for i in predicted_set])
        golden_set = set(['-'.join(i.split('-')[0: 3]) for i in golden_set])

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Aspect\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1

    def score_opinion(self, predicted_set, golden_set):
        predicted_set = set([i.split('-')[0] + '-' + ('-'.join(i.split('-')[3: 5])) for i in predicted_set])
        golden_set = set([i.split('-')[0] + '-' + ('-'.join(i.split('-')[3: 5])) for i in golden_set])

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # self.logging('Opinion\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(precision, recall, f1))
        return precision, recall, f1

    def score_triplets_intensity(self, predicted_set, golden_set):
        # 对 predicted_set 进行处理：取 0:5 和 6:8 的组合
        predicted_set = set([('-'.join(i.split('-')[0:5])) + '-' + ('-'.join(i.split('-')[6:8])) for i in predicted_set])

        # 对 golden_set 进行处理：取 0:5 和 6:8 的组合
        golden_set = set([('-'.join(i.split('-')[0:5])) + '-' + ('-'.join(i.split('-')[6:8])) for i in golden_set])

        # 获取 predicted_set 的前 5 个元素的 ID
        predicted_matched = sorted(list(predicted_set))[:5]
        predicted_ids = [i.split('-')[0] for i in predicted_matched]

        # 根据 predicted_ids 从 golden_set 中找到匹配的内容
        golden_matched = [item for item in golden_set if item.split('-')[0] in predicted_ids]

        # 打印结果
        print("Predict Matched (前 5 个):", predicted_matched)
        print("Golden Matched (匹配的内容):", golden_matched)

        # 确定正确匹配的数量
        correct_num = len(golden_set & predicted_set)

        # Precision: 正确预测的占比
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0

        # Recall: 真实标签中正确预测的占比
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0

        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1


