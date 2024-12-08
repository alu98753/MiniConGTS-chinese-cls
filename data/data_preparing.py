import math
import torch
import json
import os
import copy
import torch.nn.functional as F
from transformers import BertTokenizer
import logging
import numpy as np

class Instance(object):
    '''
    Re-organiztion for a single sentence;
    Input is in the formulation of: 
        {
        'id': '3547',
        'sentence': 'Taj Mahal offeres gret value and great food .',
        'triples': [
                    {'uid': '3547-0',
                    'target_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\B and\\O great\\O food\\O .\\O',
                    'opinion_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\O and\\O great\\B food\\O .\\O',
                    'sentiment': 'positive'},
                    {'uid': '3547-1',
                    'target_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\O and\\O great\\O food\\B .\\O',
                    'opinion_tags': 'Taj\\O Mahal\\O offeres\\O gret\\O value\\O and\\O great\\B food\\O .\\O',
                    'sentiment': 'positive'}
                    ]
        }
    Usage example:
    # sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    # instances = load_data_instances(sentence_packs, args)
    # testset = DataIterator(instances, args)
    '''
    
    def __init__(self, tokenizer, single_sentence_pack, args):
        self.args = args
        self.sentence = single_sentence_pack['sentence']
        # self.tokens = tokenizer.tokenize(self.sentence, add_prefix_space=True)  # ['ĠL', 'arg', 'est', 'Ġand', 'Ġfres', 'hest', 'Ġpieces', 'Ġof', 'Ġsushi', 'Ġ,', 'Ġand', 'Ġdelicious', 'Ġ!']
        self.tokens = tokenizer.tokenize(self.sentence)  # ['ĠL', 'arg', 'est', 'Ġand', 'Ġfres', 'hest', 'Ġpieces', 'Ġof', 'Ġsushi', 'Ġ,', 'Ġand', 'Ġdelicious', 'Ġ!']
        self.L_token = len(self.tokens)
        self.word_spans = self.get_word_spans()  # '[[0, 2], [3, 3], [4, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]]'
        self._word_spans = copy.deepcopy(self.word_spans)
        self.id = single_sentence_pack['id']

        self.triplets = single_sentence_pack['triples']
        
        # 在 `Instance.__init__` 中
        # for triplet in self.triplets:
        #     print(f"Before alignment - aspect_tags: {triplet['target_tags']}")
        #     print(f"Before alignment - opinion_tags: {triplet['opinion_tags']}")
        #     triplet['target_tags'] = align_tags_with_tokens(self.tokens, triplet['target_tags'])
        #     triplet['opinion_tags'] = align_tags_with_tokens(self.tokens, triplet['opinion_tags'])

        #     print(f"After alignment - aspect_tags: {triplet['target_tags']}")
        #     print(f"After alignment - opinion_tags: {triplet['opinion_tags']}")

        #     # 確保對齊後的標籤長度正確
        #     assert len(triplet['target_tags'].split("\\")) == self.L_token, \
        #         f"Target tags 長度與 tokens 不一致: {triplet['target_tags']} vs {self.tokens}"
        #     assert len(triplet['opinion_tags'].split("\\")) == self.L_token, \
        #         f"Opinion tags 長度與 tokens 不一致: {triplet['opinion_tags']} vs {self.tokens}"

        # 提取 intensity 標籤
        self.intensity = self.get_intensity(single_sentence_pack)

        self.triplets_in_spans = self.get_triplets_in_spans()

        self.token_classes = self.get_token_classes()

        self.cl_mask = self.get_cl_mask()
        # print(f"Tokenized: {self.tokens}")
        # print(f"Word spans: {self.word_spans}")
        # 確保 token 與 spans 一致，避免硬性依賴 split(' ')
        # assert len(self.tokens) == len(self.word_spans), \
        #     f"Tokenized length ({len(self.tokens)}) and word spans length ({len(self.word_spans)}) do not match. Sentence: {self.sentence}Tokenized length ({self.tokens}) and word spans length ({self.word_spans}) do not match. Sentence: {self.sentence}"
        # assert len(self.sentence.strip().split(' ')) == len(self.word_spans)
        # assert len(self.sentence) == len(self.word_spans)
        # print(f"原始句子: {self.sentence}")
        # print(f"標記化結果: {self.tokens}")
        # print(f"計算的字範圍: {self.word_spans}")

        # self.bert_tokens = tokenizer.encode(self.sentence, add_special_tokens=False, add_prefix_space=True)
        self.bert_tokens = tokenizer.encode(self.sentence, add_special_tokens=False)
        self.bert_tokens_padded = torch.zeros(args.max_sequence_len).long()

        self.mask = self.get_mask()

        if len(self.bert_tokens) != self._word_spans[-1][-1] + 1:
            # print("len(self.bert_tokens) != self._word_spans[-1][-1] + 1:",self.sentence, self._word_spans)
            pass
            
        for i in range(len(self.bert_tokens)):
            self.bert_tokens_padded[i] = self.bert_tokens[i]
        
        self.tagging_matrix = self.get_tagging_matrix()
        self.tagging_matrix = (self.tagging_matrix + self.mask - torch.tensor(1)).long() # 表示-1是無效位置
        self.valence_arousal_tagging_matrix = self.get_valence_arousal_tagging_matrix()
        # self.intensity_tagging_matrix = (self.intensity_tagging_matrix + self.mask - torch.tensor(1)).long()


    def get_intensity(self, single_sentence_pack):
        # 提取 intensity 並轉換為 [valence, arousal]
        # print()
        intensity = []
        for triplet in single_sentence_pack.get("triples", []):
            # print(triplet)
            intensity_values = triplet.get("intensity", "5.0#5.0")
            valence, arousal = map(float, intensity_values.split("#"))
            intensity.append([valence, arousal])
        # print("調試點3 Original Intensity Values:", intensity)  # 調試點3
        intensity_tensor = torch.tensor(intensity, dtype=torch.float32)
        # print("調試點4 Converted Intensity Tensor:", intensity_tensor)  # 調試點4
        # print(f"調試 ID: {single_sentence_pack['id']} | Intensity Tensor: {intensity_tensor}")

        return intensity_tensor

    def get_mask(self):
        mask = torch.ones((self.args.max_sequence_len, self.args.max_sequence_len))
        # print(f"max_sequence_len: {self.args.max_sequence_len}, bert_tokens_len: {len(self.bert_tokens)}")

        mask[:, len(self.bert_tokens):] = 0
        mask[len(self.bert_tokens):, :] = 0
        for i in range(len(self.bert_tokens)):
            mask[i][i] = 0
        return mask
        
    def get_word_spans(self):
        """
        根據標記化後的 token 長度，生成 word_spans。
        """
        word_spans = []
        start_idx = 0
        for token in self.tokens:
            # 每個 token 視為一個單獨的字
            end_idx = start_idx + len(token) - 1
            word_spans.append([start_idx, end_idx])
            start_idx = end_idx + 1
        # print(f"標記化結果與範圍對應: {list(zip(self.tokens, word_spans))}")

        return word_spans


    @staticmethod
    def validate_BIO_tags(tags):
        tags = tags.strip().split("\\")
        previous = "O"
        for tag in tags:
            if tag.endswith("I") and previous not in {"B", "I"}:
                return False
            previous = tag
        return True 

    # def get_triplets_in_spans(self):
    #     triplets_in_spans = []
    #     intensities = []

    #     for triplet in self.triplets:
    #         # if not self.validate_BIO_tags(triplet['target_tags']):
    #         #     print(f"Invalid target_tags: {triplet['target_tags']}")
    #         # if not self.validate_BIO_tags(triplet['opinion_tags']):
    #         #     print(f"Invalid opinion_tags: {triplet['opinion_tags']}")

    #         sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}

    #         aspect_tags = triplet['target_tags']
    #         opinion_tags = triplet['opinion_tags']
    #         sentiment = triplet['sentiment']
    #         intensity = triplet['intensity']
    #         valence, arousal = map(float, intensity.split("#"))

    #         # 將每個三元組的 intensity 與對應的 aspect_spans 和 opinion_spans 一起存儲
    #         # intensity_values = triplet.get("intensity", "5.0#5.0")
    #         # valence, arousal = map(float, intensity_values.split("#"))

    #         aspect_spans = self.get_spans_from_BIO(aspect_tags)
    #         opinion_spans = self.get_spans_from_BIO(opinion_tags)
            
    #         # print(f"原始句子: {self.sentence}")
    #         # print(f"intensity_values: {valence},{arousal}")
    #         # print(f"標記化結果: {self.tokens}")
    #         # print(f"Ltoken: {len(self.tokens)}")
    #         # print(f"aspect_tags:{aspect_tags}")
    #         # print(f"opinion_tags:{opinion_tags}")
    #         # print(f"sentiment:{sentiment}")
    #         # print(f"aspect_spans:{aspect_spans}")
    #         # print(f"opinion_spans:{opinion_spans}")
            
    #         # triplets_in_spans.append((aspect_spans, opinion_spans, sentiment2id[sentiment]))
    #         # 存儲三元組及對應的 intensity
    #         triplets_in_spans.append((aspect_spans, opinion_spans, sentiment2id[sentiment],intensity))
    #         # intensities.append([valence, arousal])
    #     # 保存 intensities 作為張量



    #     # self.intensities = torch.tensor(intensities, dtype=torch.float32)

    #     return triplets_in_spans
    
    
    def get_triplets_in_spans(self):
        triplets_in_spans = []
        for triplet in self.triplets:
            sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}

            aspect_tags = triplet['target_tags']
            opinion_tags = triplet['opinion_tags']
            sentiment = triplet['sentiment']

            intensity = triplet['intensity']  # 假設每個三元組都包含 Valence 值
            
            aspect_spans = self.get_spans_from_BIO(aspect_tags)
            opinion_spans = self.get_spans_from_BIO(opinion_tags)

            # 提取 intensity 中的 v 和 a 值
            valence_str, arousal_str = intensity.split('#')

            valence = int(round(float(valence_str)))  # 四捨五入並轉為整數 
            arousal = int(round(float(arousal_str)))   # 四捨五入並轉為整數
            # print(f"id : {triplet['uid']}")
            # print(f"valence : {valence},arousal : {arousal}")
            
            if valence >8 :
                valence =8
            if arousal >8 :
                arousal =8
            if valence < 4:
                valence = 4
            if arousal <4 :
                arousal = 4

            valence = valence-3 # 設定 Valence 值 (4~8)-3 
            arousal = arousal-3
            triplets_in_spans.append((aspect_spans, opinion_spans, sentiment2id[sentiment], valence, arousal))

        return triplets_in_spans

    def get_spans_from_BIO(self, tags):
        def is_english_letter(char):
            """判斷是否為英文字母"""
            return 'a' <= char <= 'z' or 'A' <= char <= 'Z'

        """
        解析 BIO 標籤字符串並生成 spans，處理連續數字和連續英文的情況，排除空白部分。
        :param tags: BIO 標籤字符串 (以 "\\" 分割)
        :return: span 列表 (start, end)
        """
        tags = tags.split("\\")  # 將標籤字符串分割成列表
        tags_word = tags[::2]  # 提取字詞部分
        tags_BIO = tags[1::2]  # 提取 BIO 標籤部分
        # print(f"Processing tags_word: {tags_word}")
        # print(f"Processing tags_BIO: {tags_BIO}")

        # 合併連續數字或連續英文的標籤，排除空白部分
        merged_words = []
        merged_BIO = []
        i = 0
        while i < len(tags_word):
            if tags_word[i].isdigit() or is_english_letter(tags_word[i]):  # 處理連續數字或連續英文
                # 合併連續的數字或連續英文，排除空白部分
                start = i
                while i + 1 < len(tags_word) and (tags_word[i + 1].isdigit() or is_english_letter(tags_word[i + 1]) or tags_word[i + 1] == "'"):
                    i += 1
                # 如果中間有空白，直接跳過該空白部分
                word = "".join([tags_word[j] for j in range(start, i + 1) if tags_word[j] != " "])
                merged_words.append(word)  # 合併
                merged_BIO.append(tags_BIO[start])  # 使用第一個詞的 BIO 標籤

            else:
                merged_words.append(tags_word[i])
                merged_BIO.append(tags_BIO[i])
            i += 1

        # print(f"Merged tags_word: {merged_words}")
        # print(f"Merged tags_BIO: {merged_BIO}")

        # 生成 spans
        spans = []
        start = -1  # 初始化起始點為 -1

        for i, tag in enumerate(merged_BIO):
            if tag == 'B':  # 標籤為 'B' 時
                if start != -1:  # 如果之前有未結束的 span，則先結束它
                    spans.append([start, i - 1])
                start = i  # 設置新 span 的開始
            elif tag == 'I':  # 標籤為 'I' 時，維持 span 的延續
                if start == -1:  # 如果 `I` 沒有緊接在 `B` 之後，忽略
                    continue
            else:  # 遇到 'O' 或其他標記時，結束當前 span
                if start != -1:  # 如果有未結束的 span，則結束它
                    spans.append([start, i - 1])
                    start = -1  # 重置 start

        # 如果字符串結束時有未結束的 span，將其補充
        if start != -1:
            spans.append([start, len(merged_BIO) - 1])

        # print(f"Generated spans: {spans}")
        return spans

    def get_token_classes(self):
        # 0: NULL
        # 1: Aspect
        # 2: Opinion negative
        # 3: Opinion neutral
        # 4: Opinion positive
        token_classes = [0] * self.L_token
        # sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}
        for aspect_spans, opinion_spans, sentiment ,_,_ in self.triplets_in_spans:
            # print(f"調試：Aspect span={aspect_spans}, Opinion span={opinion_spans}, Sentiment={sentiment}")
            for a in aspect_spans:
                _a = copy.deepcopy(a)
                token_classes[_a[0]: _a[-1]+1] = [1] * (_a[-1]+1 - _a[0])
            for o in opinion_spans:
                _o = copy.deepcopy(o)
                token_classes[_o[0]: _o[-1]+1] = [sentiment] * (_o[-1]+1 - _o[0])
        # print(f"token_classes:{token_classes}")
        return token_classes

    def get_cl_mask(self):
        assert len(self.token_classes) == self.L_token, f"token_classes長度 ({len(self.token_classes)}) 與L_token ({self.L_token}) 不一致。token_classes ({self.token_classes}) 與L_token ({self.tokens})"

        token_classes = torch.tensor(self.token_classes).unsqueeze(0).expand(self.L_token, -1)
        eq = (token_classes == token_classes.T)
        mask01 = ((torch.tril(torch.ones(self.L_token, self.L_token)) - 1) * (-1))
        m = (eq * 2 - 1) * mask01
        pad_len = self.args.max_sequence_len - self.L_token
        return F.pad(m, (0, pad_len, 0, pad_len), "constant", 0)
    
    
    # 設定 logging
    # logging.basicConfig(filename="debug_get_tagging_matrix.log", 
    #                     level=logging.DEBUG, 
    #                     format="%(asctime)s - %(message)s")    
    def get_tagging_matrix(self):
        '''
        mapping the tags to a Matrix Tagginh scheme
        '''
        tagging_matrix = torch.zeros((self.args.max_sequence_len, self.args.max_sequence_len))
        '''
        tagging_matrix                      O   
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        A      [0., 0., 0., 0., 0., 0., 0., S , 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        '''
        # logging.debug("Start get_tagging_matrix")
        # logging.debug(f"Sentence: {self.sentence}")
        # logging.debug(f"Tokenized: {self.tokens}")
        # logging.debug(f"L_token: {self.L_token}")
        for triplet in self.triplets_in_spans:
        # for idx, triplet in enumerate(self.triplets_in_spans):

            # triplets = [{'uid': '2125-0',
            #   'target_tags': 'Largest\\O and\\O freshest\\O pieces\\B of\\I sushi\\I ,\\O and\\O delicious\\O !\\O',
            #   'opinion_tags': 'Largest\\B and\\O freshest\\B pieces\\O of\\O sushi\\O ,\\O and\\O delicious\\B !\\O',
            #   'sentiment': 'positive'}]

            # print(aspect_tags)
            # print(opinion_tags)
            # print(sentiment_tags)
            # Largest\O and\O freshest\O pieces\B of\I sushi\I ,\O and\O delicious\O !\O
            # Largest\B and\O freshest\B pieces\O of\O sushi\O ,\O and\O delicious\B !\O
            # positive
            # break
            sentiment2id = {'negative': 2, 'neutral': 3, 'positive': 4}
            
            # if len(self.triplets_in_spans) != 3:
            #     print(self.triplets_in_spans)

            aspect_spans, opinion_spans, sentiment ,_ ,_ = triplet
            # Logging 當前的 triplet 資訊
            # logging.debug(f"Triplet {idx}: Aspect spans={aspect_spans}, Opinion spans={opinion_spans}, Sentiment={sentiment}")

            # aspect_spans = self.get_spans_from_BIO(aspect_tags, self.word_spans)
            # opinion_spans = self.get_spans_from_BIO(opinion_tags, self.word_spans)

            '''set tag for sentiment'''
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    # print(aspect_span)
                    # print(opinion_span)
                    al = aspect_span[0]
                    ar = aspect_span[1]
                    pl = opinion_span[0]
                    pr = opinion_span[1]
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            # print(al, ar, pl, pr)
                            # print(i, j)
                            # print(i==al and j==pl)
                            if i==al and j==pl:
                                tagging_matrix[i][j] = sentiment  # 3 4 5
                                # logging.debug(f"Set tagging_matrix[{i}][{j}] = {sentiment}")

                            else:
                                tagging_matrix[i][j] = 1  # 1: ctd
                                # logging.debug(f"Set tagging_matrix[{i}][{j}] = 1")
        
        # logging.debug(f"Completed tagging_matrix for sentence: {self.sentence}")
        # print(f"tagging_matrix shape = {tagging_matrix.shape}")

        return tagging_matrix

    def get_valence_arousal_tagging_matrix(self):
        '''
        mapping the tags to two separate matrices: Valence Matrix (M_V) and Arousal Matrix (M_A)
        '''
        # 初始化矩陣，所有位置初始設為 MSK 類別
        valence_matrix = torch.full((self.args.max_sequence_len, self.args.max_sequence_len), fill_value=-1, dtype=torch.long)
        arousal_matrix = torch.full((self.args.max_sequence_len, self.args.max_sequence_len), fill_value=-1, dtype=torch.long)
    
        for triplet in self.triplets_in_spans:
            aspect_spans, opinion_spans,_, valence, arousal = triplet
            
            # Set the valence and arousal tags for each Aspect-Opinion pair
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    al = aspect_span[0]
                    ar = aspect_span[1]
                    pl = opinion_span[0]
                    pr = opinion_span[1]
                    for i in range(al, ar + 1):
                        for j in range(pl, pr + 1):
                            # if i == al and j == pl:
                            valence_matrix[i][j] = valence  
                            arousal_matrix[i][j] = arousal  
                            # else:
                            #     valence_matrix[i][j] = -1  # 非主要 Aspect-Opinion 組合位置
                            #     arousal_matrix[i][j] = -1

        return valence_matrix, arousal_matrix



class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    # DataIterator需要將intensity標籤加入batch返回值...

    def get_batch(self, index):
        sentence_ids = []
        word_spans = []
        bert_tokens = []
        masks = []
        tagging_matrices = []
        tokenized = []
        cl_masks = []
        token_classes = []
        # 原有代碼...
        max_triplets = 0
        valence_matrices = []
        arousal_matrices = []

        for i in range(index * self.args.batch_size, min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            word_spans.append(self.instances[i].word_spans)
            bert_tokens.append(self.instances[i].bert_tokens_padded)
            masks.append(self.instances[i].mask)
            # aspect_tags.append(self.instances[i].aspect_tags)
            # opinion_tags.append(self.instances[i].opinion_tags)
            tagging_matrices.append(self.instances[i].tagging_matrix)
            tokenized.append(self.instances[i].tokens)
            cl_masks.append(self.instances[i].cl_mask)
            token_classes.append(self.instances[i].token_classes)

            # 收集 intensity，並更新批次內的最大三元組數量
            # 獲取 Valence 和 Arousal 矩陣
            valence_matrix, arousal_matrix = self.instances[i].get_valence_arousal_tagging_matrix()
            valence_matrices.append(valence_matrix)
            arousal_matrices.append(arousal_matrix)


        # 將數據處理為張量並返回
        if len(bert_tokens) == 0:
            print(bert_tokens)
        # Debug: Check padded intensities
        # print(f"Padded intensities shape: {intensities.shape}")
        # print(f"Padded intensities content: {intensities}")

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        # lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        # aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        # opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tagging_matrices = torch.stack(tagging_matrices).long().to(self.args.device)
        cl_masks = torch.stack(cl_masks).long().to(self.args.device)
        
        # Debug: Verify tensor shapes and devices
        # print(f"Bert tokens shape: {bert_tokens.shape}, device: {bert_tokens.device}")
        # print(f"Masks shape: {masks.shape}, device: {masks.device}")
        # print(f"Tagging matrices shape: {tagging_matrices.shape}, device: {tagging_matrices.device}")
        # print(f"CL masks shape: {cl_masks.shape}, device: {cl_masks.device}")
        
        # print(f"intensity_tagging_matrices shape: {intensity_tagging_matrices.shape}") 16 80 80 2
        valence_matrices = torch.stack(valence_matrices).long().to(self.args.device)
        arousal_matrices = torch.stack(arousal_matrices).long().to(self.args.device)

        # return sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes
        return sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes , valence_matrices, arousal_matrices


if __name__ == "__main__":

    import sys
    from transformers import RobertaTokenizer, RobertaModel
    import argparse
    # import os
    
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    
    # 将上一级目录添加到sys.path
    sys.path.append(parent_dir)
    from utils.data_utils import load_data_instances

    # Load Dataset
    train_sentence_packs = json.load(open(os.path.abspath('D1/res14//NYCU_train.json')))#'D1/res14/train.json'
    # # random.shuffle(train_sentence_packs)
    # dev_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/dev.json')))
    # test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/test.json')))


    #加载预训练字典和分词方法
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
    #     cache_dir="../modules/models/",  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
    #     force_download=False,  # 是否强制下载
    # )

    # 创建一个TensorBoard写入器
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_len', type=int, default=100, help='max length of the tagging matrix')
    parser.add_argument('--sentiment2id', type=dict, default={'negative': 2, 'neutral': 3, 'positive': 4}, help='mapping sentiments to ids')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/', help='model cache path')
    # parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='reberta model path')
    parser.add_argument('--model_name_or_path', type=str, default='hfl/chinese-roberta-wwm-ext', help='reberta model path')
    parser.add_argument('--batch_size', type=int, default=16, help='json data path')
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')
    parser.add_argument('--prefix', type=str, default="./data/", help='dataset and embedding path prefix')

    parser.add_argument('--data_version', type=str, default="D1", choices=["D1", "D2"], help='dataset and embedding path prefix')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"], help='dataset')

    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument('--epochs', type=int, default=2000, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=5, help='label number')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"], help='option: pair, triplet')
    parser.add_argument('--model_save_dir', type=str, default="./modules/models/saved_models/", help='model path prefix')
    parser.add_argument('--log_path', type=str, default="log.log", help='log path')


    args = parser.parse_known_args()[0]

    
    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    # dev_instances = load_data_instances(tokenizer, dev_sentence_packs, args)
    # test_instances = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    # devset = DataIterator(dev_instances, args)
    # testset = DataIterator(test_instances, args)
