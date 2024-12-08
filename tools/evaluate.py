import torch
import torch.nn.functional as F
from utils.common_utils import Logging
from tools.metric import Metric

from utils.eval_utils import get_triplets_set
import numpy as np


def evaluate(model, dataset, stop_words, logging, args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        # all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        all_tokenized = []
        all_valence_true = []
        all_arousal_true = []
        all_valence_preds = []
        all_arousal_preds = []
        
        for i in range(dataset.batch_count):                                                        #, batch_mean, batch_std 
            sentence_ids, tokens, masks, token_ranges, tags, tokenized, _, _ , valence_matrices, arousal_matrices= dataset.get_batch(i)
            # sentence_ids, bert_tokens, masks, word_spans, tagging_matrices = trainset.get_batch(i)
            # print(f"tags{i}:{tags}")
            preds, _, _,  logits_valence, logits_arousal = model(tokens, masks) 

            # print(f"調試點9 Batch {i} True Intensities:", intensities)  # 調試點9
            # print(f"調試點10 Batch {i} Predicted Intensities:", intensity_pred)  # 調試點10
            preds = torch.argmax(preds, dim=3) #2
            # print(f"preds shape: {preds.shape}")
            all_preds.append(preds) #3
            all_labels.append(tags) #4
            # all_lengths.append(lengths) #5
            sens_lens = [len(token_range) for token_range in token_ranges]
            all_sens_lengths.extend(sens_lens) #6
            all_token_ranges.extend(token_ranges) #7
            all_ids.extend(sentence_ids) #8
            all_tokenized.extend(tokenized)
            # intensity 處 
            # print(f"logits_valence:{logits_valence.shape}")

            # Step 1: Get the predicted class with the highest probability
            valence_prob = F.softmax(logits_valence, dim=-1)
            arousal_pred = F.softmax(logits_arousal, dim=-1)

            valence_preds = torch.argmax(valence_prob, dim=3)
            arousal_preds = torch.argmax(arousal_pred, dim=3)

            # Step 2:Store the actual predicted Valence matrix for evaluation
            all_valence_true.append(valence_matrices)
            all_arousal_true.append(arousal_matrices)
            all_valence_preds.append(valence_preds)
            all_arousal_preds.append(arousal_preds)

            # print(f"Batch {i} intensities shape: {intensities.shape}")
            # print(f"Batch {i} sentence_ids length: {len(sentence_ids)}")
        # print(f"Total samples collected: {len(all_ids)}")
        # print(f"Total intensities collected: {len(all_intensities)}")
        
        # print(f"Shape of all_intensities: {len(all_intensities)}, Type: {type(all_intensities)}")
        # print(f"Shape of all_intensity_pred: {len(all_intensity_pred)}, Type: {type(all_intensity_pred)}")

        
        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_valence_true = torch.cat(all_valence_true, dim=0).cpu().tolist()
        all_arousal_true = torch.cat(all_arousal_true, dim=0).cpu().tolist()        # all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        all_valence_preds = torch.cat(all_valence_preds, dim=0).cpu().tolist()
        all_arousal_preds = torch.cat(all_arousal_preds, dim=0).cpu().tolist()        # all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        print(f"調試 type(all_valence_true)：{type(all_valence_true)}")  # 取第一個元素
        print(f"調試 shape(all_valence_true)：{np.shape(all_valence_true)}")  # 取第一個元素


        # 引入 metric 计算评价指标
        # metric = Metric(args, stop_words, all_tokenized, all_ids, all_preds, all_labels, all_sens_lengths, all_token_ranges, ignore_index=-1, logging=logging)
                # 引入 Metric 並傳遞 intensity 數據
        metric = Metric(args, stop_words, all_tokenized, all_ids, all_preds, all_labels, 
                        all_sens_lengths, all_token_ranges, 
                        all_valence_true, all_arousal_true, all_valence_preds, all_arousal_preds, logging=logging)

        _ ,predicted_set, golden_set = metric.get_sets()
        # for i in range(5):
        # Call the method to extract and print triplets
        # metric.extract_and_print_triplets(p_predicted_set)
        
        print(f"調試predicted_set length：{len(predicted_set)},調試golden_set length：{len(golden_set)}")
        # print(f"調試predicted_set：{list(predicted_set)[:5]}")  # 取第一個元素
        # print(f"調試golden_set：{list(golden_set)[:5]}")        

        predicted_id_prefixes = [elem.split('-')[0] for elem in list(predicted_set)[:5]]
        golden_set_5 = []

        for i in predicted_id_prefixes:
            # print(f"predicted id prefixes: {i}")
            for j in golden_set:
                # print(f"Predicted element: {j}")
                if isinstance(j, str):  # Ensure j is a string
                    # print(f"{j.split('-')[0]}/////{i}")
                    if j.split('-')[0] == i:
                        golden_set_5.append(j)  # Correct usage of append
                else:
                    # print(f"Unexpected type in predicted_set: {type(j)}")
                    pass

        # print(f"調試golden_set: {golden_set_5[:5]}")
            
        
        aspect_results = metric.score_aspect(predicted_set, golden_set)
        opinion_results = metric.score_opinion(predicted_set, golden_set)
        pair_results = metric.score_pairs(predicted_set, golden_set)
        
        precision, recall, f1 = metric.score_triplets(predicted_set, golden_set)

        # 計算 Triplet_intensity 的指標
        # 調試：打印傳入的 Intensity 值
        # print(f"調試：intensity_logits.shape = {intensity_pred.shape}, intensities.shape = {intensities.shape}")
        triplet_intensity_precision, triplet_intensity_recall, triplet_intensity_f1 = metric.score_triplets_intensity(predicted_set, golden_set)
        
        aspect_results = [100 * i for i in aspect_results]
        opinion_results = [100 * i for i in opinion_results]
        pair_results = [100 * i for i in pair_results]

        triplet_intensity_precision = 100 * triplet_intensity_precision
        triplet_intensity_recall = 100 * triplet_intensity_recall
        triplet_intensity_f1 = 100 * triplet_intensity_f1
        precision = 100 * precision
        recall = 100 * recall
        f1 = 100 * f1        
        logging('Aspect\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(aspect_results[0], aspect_results[1], aspect_results[2]))
        logging('Opinion\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(opinion_results[0], opinion_results[1], opinion_results[2]))
        logging('Pair\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(pair_results[0], pair_results[1], pair_results[2]))
        logging('Triplet\tP:{:.2f}\tR:{:.2f}\tF1:{:.2f}'.format(precision, recall, f1))
        # 將結果打印到 log
        logging(f'Triplet_intensity P:{triplet_intensity_precision:.2f} R:{triplet_intensity_recall:.2f} F1:{triplet_intensity_f1:.2f}')
    
    model.train()
    return precision, recall, f1 , pair_results[2] ,triplet_intensity_f1
    # return 0, 0, 0
