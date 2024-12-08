

import json
from transformers import RobertaTokenizer, RobertaModel
# from transformers import AutoTokenizer, RobertaModel
import numpy as np
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import trange
import datetime
import os, random
from utils.common_utils import Logging

from utils.data_utils import load_data_instances
from data.data_preparing import DataIterator
from torch.optim.lr_scheduler import ReduceLROnPlateau


from modules.models.roberta import Model
from modules.f_loss import FocalLoss

from tools.trainer import Trainer
from tools.evaluate import evaluate
from tools.metric import Metric

from utils.common_utils import stop_words
from transformers import BertTokenizer
from data.data_preparing import Instance



if __name__ == '__main__':
    torch.cuda.set_device(0)
    print(f"the GPU use now: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_len', type=int, default=100, help='max length of the tagging matrix')
    parser.add_argument('--sentiment2id', type=dict, default={'negative': 2, 'neutral': 3, 'positive': 4}, help='mapping sentiments to ids')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/', help='model cache path')
    parser.add_argument('--model_name_or_path', type=str, default='hfl/chinese-roberta-wwm-ext', help='reberta model path')
    # parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='reberta model path')
    parser.add_argument('--batch_size', type=int, default=16, help='json data path')
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')
    parser.add_argument('--prefix', type=str, default="./data/", help='dataset and embedding path prefix')

    parser.add_argument('--data_version', type=str, default="D1", choices=["D1", "D2"], help='dataset and embedding path prefix')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"], help='dataset')

    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument('--epochs', type=int, default=2000, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=5, help='label number')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"], help='option: pair, triplet')
    parser.add_argument('--model_save_dir', type=str, default="/mnt/md0/chen-wei/zi/MiniConGTS_copy_ch_cantrain/modules/models/saved_models/", help='model path prefix')
    parser.add_argument('--log_path', type=str, default="/mnt/md0/chen-wei/zi/MiniConGTS-chinese-cls_2/log/", help='log path')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], help='模式：train 或 predict')
    parser.add_argument('--input_file', type=str, default='/mnt/md0/chen-wei/zi/MiniConGTS-chinese-cls_2/data/D1/res14/NYCU_NLP_113A_Test.txt', help='预测模式下的输入文件')
    parser.add_argument('--output_file', type=str, default='submission.txt', help='预测结果输出文件')


    args = parser.parse_known_args()[0]
    # if args.log_path is None:
    args.log_path = '{}log_{}_{}_{}.log'.format(args.log_path ,args.data_version, args.dataset, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
#/mnt/md0/chen-wei/zi/MiniConGTS_copy/log/
    #加载预训练字典和分词方法
    # tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path,
    #     cache_dir=args.model_cache_dir,  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
    #     force_download=False,  # 是否强制下载
    # )

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # tokenizer = RobertaModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    logging = Logging(file_name=args.log_path).logging


    def seed_torch(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
    seed = 666
    seed_torch(seed)
    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logging(f"""
            \n\n
            ========= - * - =========
            date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            seed: {seed}
            ========= - * - =========
            """
        )


    # Load Dataset
    # train_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/train.json')))NYCU_train.json
    train_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/NYCU/NYCU_train.json'), encoding='utf-8')) #train_with_intensity.json
    
    # random.shuffle(train_sentence_packs)
    # dev_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/NYCU_dev.json'), encoding='utf-8'))#dev_with_intensity.json
    test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/NYCU/NYCU_test.json'), encoding='utf-8'))#test_with_intensity.json

    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    # dev_instances = load_data_instances(tokenizer, dev_sentence_packs, args)
    test_instances = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    # devset = DataIterator(dev_instances, args)
    devset = DataIterator(test_instances, args)
    testset = DataIterator(test_instances, args)


    if torch.cuda.is_available():
        device = torch.device(args.device)
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    model = Model(args).to(args.device)
    optimizer = torch.optim.AdamW([
                    {'params': model.bert.parameters(), 'lr': 1e-5},
                    {'params': model.linear1.parameters(), 'lr': 1e-2},
                    {'params': model.cls_linear.parameters(), 'lr': 1e-3},
                    {'params': model.cls_linear1.parameters(), 'lr': 1e-3},
                    {'params': model.cls_linear_valence.parameters(), 'lr': 1e-1, 'weight_decay': 1e-4}, 
                    {'params': model.cls_linear_arousal.parameters(), 'lr': 1e-1, 'weight_decay': 1e-4}], lr=1e-3)#SGD, momentum=0.9  
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 600, 1000], gamma=0.5, verbose=True)
    # valence 和 arousal 的 ReduceLROnPlateau
    lr_scheduler_valence = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=1e-5, min_lr=1e-4, verbose=True)
    lr_scheduler_arousal = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=1e-5, min_lr=1e-4, verbose=True)


    # label = ['N', 'CTD', 'POS', 'NEU', 'NEG']
    weight = torch.tensor([1.0, 6.0, 6.0, 6.0, 6.0]).float().cuda()
    f_loss = FocalLoss(weight, ignore_index=-1)#.forwardf_loss(preds,labels)

    weight1 = torch.tensor([1.0, 4.0]).float().cuda()
    f_loss1 = FocalLoss(weight1, ignore_index=-1)

    # weight2 = torch.tensor([1.0, 6.73, 12.30, 3.0, 5.60, 45.05]).float().cuda()
    weight2 = torch.tensor([1.0, 6.0, 10.0, 3.0, 6.0, 10]).float().cuda()
    f_loss2 = FocalLoss(weight2, ignore_index=-1)#.forwardf_loss(preds,labels)

    # weight3 = torch.tensor([1.0, 45.0, 4.0, 2.0, 10.31, 50.50]).float().cuda()
    weight3 = torch.tensor([1.0, 10.0, 6.0, 3.0, 10.0, 10.0]).float().cuda()
    f_loss3 = FocalLoss(weight3, ignore_index=-1)

    '''
    Intensity ������� (v):
    Intensity: 4, ���: 14.86%
    Intensity: 5, ���: 8.13%
    Intensity: 6, ���: 56.55%
    Intensity: 7, ���: 17.86%
    Intensity: 8, ���: 0.84%

    Intensity ������� (a):
    Intensity: 4, ���: 1.82%
    Intensity: 5, ���: 31.58%
    Intensity: 6, ���: 55.77%
    Intensity: 7, ���: 9.70%
    Intensity: 8, ���: 1.13%
'''

    '''
    ## beta_1 和 beta_2 : weight of loss function 
    # 
    # 用於平衡loss1（次級損失函數）和loss_cl（對比損失）在總損失中的貢獻：

    bear_max和last：

        這些參數似乎可以管理使用對比學習時的訓練行為 ( self.contrastive)。
        儘管在提供的函數中註解掉了train，但邏輯表明：
        bear_max可能代表在激活對比學習之前性能沒有提高的連續時期的最大數量。
        last似乎是一個倒數計時器或限制對比學習保持活躍的時期數：

    '''
    
    '''
    下一步
        調整參數beta_1：根據beta_2任務中每個損失函數的相對重要性來選擇值。同樣，設定bear_max並last控制訓練動態。
        取消註釋邏輯：取消方法中相關部分的註釋，以啟動由和train控制的對比學習邏輯。bear_maxlast
        實驗：使用這些參數進行實驗，觀察它們對模型表現和訓練動態的影響。    
    '''
    beta_1 = 1.0  # Weight for loss1
    beta_2 = 0.5  # Weight for contrastive loss
    bear_max = 5  # Maximum patience before enabling contrastive learning
    last = 10     # Duration for which contrastive learning remains active
    # trainer = Trainer(model, trainset, devset, testset, optimizer, (f_loss, f_loss1), lr_scheduler, args, logging)

    # # Run evaluation
    ######################
    def prepare_data_for_prediction(lines, tokenizer, args):
        ids = []
        sentences = []
        for line in lines:
            sentence_id = line[0].strip()
            sentence = line[1].strip()
            ids.append(sentence_id)
            sentences.append(sentence)

        bert_tokens_list = []
        masks_list = []

        for sent in sentences:
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            # padding
            if len(tokens) > args.max_sequence_len:
                tokens = tokens[:args.max_sequence_len]
            bert_tokens_padded = tokens + [0]*(args.max_sequence_len - len(tokens))
            
            # 建 mask: 與訓練時相同，對角線為0，其餘有效位置為1
            length = len(tokens)
            mask = torch.ones((args.max_sequence_len, args.max_sequence_len))
            mask[:, length:] = 0
            mask[length:, :] = 0
            for i in range(length):
                mask[i][i] = 0

            bert_tokens_list.append(torch.tensor(bert_tokens_padded, dtype=torch.long))
            masks_list.append(mask)

        bert_tokens_tensor = torch.stack(bert_tokens_list, dim=0).to(args.device)
        masks_tensor = torch.stack(masks_list, dim=0).to(args.device)
        
        return ids, sentences, bert_tokens_tensor, masks_tensor

    def decode_triplets_from_logits(preds, valence_preds, arousal_preds, bert_tokens, tokenizer, args):
        batch_size, L, _ = preds.shape
        results = []

        for b in range(batch_size):
            pred_mat = preds[b].cpu().numpy()
            val_mat = valence_preds[b].cpu().numpy()
            aro_mat = arousal_preds[b].cpu().numpy()

            token_ids = bert_tokens[b].cpu().numpy()
            if 0 in token_ids:
                valid_length = np.where(token_ids == 0)[0][0]
            else:
                valid_length = len(token_ids)

            tokens = tokenizer.convert_ids_to_tokens(token_ids[:valid_length])

            triplet_list = []
            for i in range(1, valid_length-1):
                for j in range(1, valid_length-1):
                    if i == j:
                        continue
                    # sentiment2id: negative=2, neutral=3, positive=4
                    if pred_mat[i][j] in [2, 3, 4]:
                        # 找 aspect span
                        a_start, a_end = i, i
                        while a_end+1 < valid_length and pred_mat[a_end+1][j] == 1:
                            a_end += 1
                        # 找 opinion span
                        o_start, o_end = j, j
                        while o_end+1 < valid_length and pred_mat[i][o_end+1] == 1:
                            o_end += 1

                        aspect_tokens = tokens[a_start:a_end+1]
                        opinion_tokens = tokens[o_start:o_end+1]

                        sub_val = val_mat[a_start:a_end+1, o_start:o_end+1]
                        sub_aro = aro_mat[a_start:a_end+1, o_start:o_end+1]

                        valid_val = sub_val[sub_val != -1]
                        valid_aro = sub_aro[sub_aro != -1]

                        if len(valid_val) > 0 and len(valid_aro) > 0:
                            v = valid_val.mean()
                            a = valid_aro.mean()
                            real_v = v + 3
                            real_a = a + 3
                            intensity_str = f"{real_v:.2f}#{real_a:.2f}"

                            aspect_text = ''.join(aspect_tokens).replace('##', '')
                            opinion_text = ''.join(opinion_tokens).replace('##', '')

                            triplet_list.append({
                                "aspect": aspect_text,
                                "opinion": opinion_text,
                                "intensity": intensity_str
                            })

            results.append(triplet_list)

        return results


    if args.mode == 'train':
        # Run train
        trainer = Trainer(model, trainset, devset, testset, optimizer, (f_loss, f_loss1,f_loss2, f_loss3), lr_scheduler, lr_scheduler_valence,lr_scheduler_arousal, args, logging, beta_1, beta_2, bear_max, last)
        trainer.train()
        model.eval()

        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(',', 1) for line in f.readlines()]

        if not all(len(line) == 2 for line in lines):
            raise ValueError("输入文件中的每一行必须包含两个逗号分隔的列：'ID, Sentence'。")

        batch_size = args.batch_size
        total_len = len(lines)
        all_ids = []
        all_sents = []
        all_results = []

        # 分批處理避免OOM
        for start_idx in range(0, total_len, batch_size):
            batch_lines = lines[start_idx: start_idx + batch_size]
            ids, raw_sentences, bert_tokens, masks = prepare_data_for_prediction(batch_lines, tokenizer, args)

            all_ids.extend(ids)
            all_sents.extend(raw_sentences)

            with torch.no_grad():
                logits, logits1, sim_matrices, logits_valence, logits_arousal = model(bert_tokens, masks)
                preds = torch.argmax(logits, dim=3)
                valence_preds = torch.argmax(F.softmax(logits_valence, dim=-1), dim=3)
                arousal_preds = torch.argmax(F.softmax(logits_arousal, dim=-1), dim=3)

            batch_triplets = decode_triplets_from_logits(preds, valence_preds, arousal_preds, bert_tokens, tokenizer, args)
            all_results.extend(batch_triplets)

        # 同一ID的triplets合併在同一行
        id2triplets = {}
        for idx, sid in enumerate(all_ids):
            triplets = all_results[idx]
            if sid not in id2triplets:
                id2triplets[sid] = []
            for t in triplets:
                trip_str = f"({t['aspect']},{t['opinion']},{t['intensity']})"
                id2triplets[sid].append(trip_str)

        with open(args.output_file, 'w', encoding='utf-8') as fw:
            for sid in id2triplets:
                if len(id2triplets[sid]) == 0:
                    # 沒有 triplets，只輸出 ID
                    fw.write(f"{sid}\n")
                else:
                    # 有 triplets，在同一行輸出
                    line_triplets = "".join(id2triplets[sid])
                    fw.write(f"{sid} {line_triplets}\n")
    elif args.mode == 'predict':
        # Load the pre-trained model
        # saved_model_path = os.path.join(args.model_save_dir, "best_model_ch_best.pt")
        saved_model_path = os.path.join(r"E:\NYCU-Project\Class\NLP\MiniConGTS-chinese\modules\models\saved_models\best_model_ch.pt")
        
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"模型文件 {saved_model_path} 未找到。")
        model = Model(args)  # 重新實例化模型
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(',', 1) for line in f.readlines()]

        if not all(len(line) == 2 for line in lines):
            raise ValueError("输入文件中的每一行必须包含两个逗号分隔的列：'ID, Sentence'。")

        batch_size = args.batch_size
        total_len = len(lines)
        all_ids = []
        all_sents = []
        all_results = []

        # 分批處理避免OOM
        for start_idx in range(0, total_len, batch_size):
            batch_lines = lines[start_idx: start_idx + batch_size]
            ids, raw_sentences, bert_tokens, masks = prepare_data_for_prediction(batch_lines, tokenizer, args)

            all_ids.extend(ids)
            all_sents.extend(raw_sentences)

            with torch.no_grad():
                logits, logits1, sim_matrices, logits_valence, logits_arousal = model(bert_tokens, masks)
                preds = torch.argmax(logits, dim=3)
                valence_preds = torch.argmax(F.softmax(logits_valence, dim=-1), dim=3)
                arousal_preds = torch.argmax(F.softmax(logits_arousal, dim=-1), dim=3)

            batch_triplets = decode_triplets_from_logits(preds, valence_preds, arousal_preds, bert_tokens, tokenizer, args)
            all_results.extend(batch_triplets)

        # 同一ID的triplets合併在同一行
        id2triplets = {}
        for idx, sid in enumerate(all_ids):
            triplets = all_results[idx]
            if sid not in id2triplets:
                id2triplets[sid] = []
            for t in triplets:
                trip_str = f"({t['aspect']},{t['opinion']},{t['intensity']})"
                id2triplets[sid].append(trip_str)

        with open(args.output_file, 'w', encoding='utf-8') as fw:
            for sid in id2triplets:
                if len(id2triplets[sid]) == 0:
                    # 沒有 triplets，只輸出 ID
                    fw.write(f"{sid}\n")
                else:
                    # 有 triplets，在同一行輸出
                    line_triplets = "".join(id2triplets[sid])
                    fw.write(f"{sid} {line_triplets}\n")