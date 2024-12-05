import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F

from utils.common_utils import stop_words
from tools.evaluate import evaluate

from tqdm import trange
from utils.plot_utils import gather_features, plot_pca, plot_pca_3d
import copy
import os 
import gc
import subprocess
import numpy as np
import math
import random

def get_gpu_temperature():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)
    return int(result.stdout.decode('utf-8').strip())


class Trainer():
    def __init__(self, model, trainset, devset, testset, optimizer, criterion, lr_scheduler, args, logging, beta_1, beta_2, bear_max, last, plot=False):
        self.model = model
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.optimizer = optimizer
        # self.criterion = criterion
        self.f_loss = criterion[0]  # sentiment損失
        self.f_loss1 = criterion[1]  # opinion損失
        self.intensity_loss_fn = nn.MSELoss()  # intensity損失

        self.lr_scheduler = lr_scheduler
        self.best_joint_f1 = 0
        self.best_joint_f1_test = 0
        self.best_joint_epoch = 0
        self.best_joint_epoch_test = 0

        self.writer = SummaryWriter()
        self.args = args
        self.logging = logging

        self.evaluate = evaluate
        self.stop_words = stop_words

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.plot = plot
        # self.bear = 0
        self.bear_max = bear_max
        self.last = last
        self.contrastive = True
      
        self.gamma = 0.001  # gamma 初始值
        self.alpha = 0.2  # 冪次增長速度參數
        self.threshold = 35  # joint_pair_f1 的臨界值
        self.wait_epochs = 1  # joint_pair_f1 需超過閾值的連續 epoch 數
        self.above_threshold_count = 0  # 初始化計數器
        
    def generate_triplet_mask(self,tagging_matrices):
        """
        生成三元組 Mask，用於過濾有效的三元組數據，包括延續位置（標籤為 1）。
        """
        # 有效標籤值（情感標籤 + 延續位置）
        valid_labels = torch.tensor([1, 3, 4, 5], device=tagging_matrices.device)
        
        # 初始 Mask 為全 False
        mask = torch.zeros_like(tagging_matrices, dtype=torch.bool)
        
        # 遍歷有效標籤，將對應位置設置為 True
        for label in valid_labels:
            mask |= (tagging_matrices == label)
        
        return mask

        
    def train(self):
        bear = 0
        last = self.last
        
        # 定義輸入模型路徑
        saved_model_path = os.path.join(r"/mnt/md0/chen-winiConGTS_ch_can/modules/models/saved_models/best_model_ch.pt")

        # 如果模型文件存在，則加載模型   
        if os.path.exists(saved_model_path):
            print(f"加載模型檔案 {saved_model_path}")
            self.model = torch.load(saved_model_path)
            self.model = self.model.to(self.args.device)
            self.model.train()  # 设置为训练模式
            

        else:
            print(f"模型檔案 {saved_model_path} 不存在，跳過加載。")
           
        # intensity_head学习率初始值
        current_intensity_lr = intensity_initial_lr = 1e-5
        intensity_lr_max = 5e-3  # 最大学习
        
        for i in range(self.args.epochs):
            
            # if bear >= self.bear_max and last > 0:
            #     self.contrastive = True

            # if self.contrastive:
            #     last -= 1
            #     if last == 0:
            #         bear = 0
            #         self.contrastive = False
            #         last = 10

            # print("epoch: ", i+1, "contrastive: ", self.contrastive, "bear/max: ", f"{bear}/{self.bear_max}", "last: ", last)   

            if self.plot:
                if i % 10 == 0:
                    model = copy.deepcopy(self.model)
                    gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4 = gather_features(model, self.testset)

                    plot_pca(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4, i)
                    plot_pca_3d(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4, i)
                    
            epoch_sum_loss = []
            joint_precision, joint_recall, joint_f1 ,joint_pair_f1= self.evaluate(self.model, self.devset, self.stop_words, self.logging, self.args)
            joint_precision_test, joint_recall_test, joint_f1_test,joint_pair_f1_test = self.evaluate(self.model, self.testset, self.stop_words, self.logging, self.args)
            # 更新計數器
            if joint_pair_f1 > self.threshold:
                self.above_threshold_count += 1
            else:
                self.above_threshold_count = 0

            # 如果計數器達到條件，逐漸增長 intensity_initial_lr
            if self.above_threshold_count >= self.wait_epochs:
                current_intensity_lr = min(intensity_initial_lr * (1 + (self.above_threshold_count - self.wait_epochs) *3), intensity_lr_max)
                self.gamma = 0.01
            else:
                current_intensity_lr = intensity_initial_lr
                self.gamma = 0.001
                
            # 更新 intensity_head 的学习率
            for param_group in self.optimizer.param_groups:
                if 'intensity_head' in str(param_group['params'][0]):
                    param_group['lr'] = current_intensity_lr

            self.logging('\n\nEpoch:{}'.format(i+1))
            self.logging(f"contrastive: {self.contrastive} | bear/max: {bear}/{self.bear_max} | last: {last}")
            self.logging(f"Epoch {i + 1}: joint_pair_f1 = {joint_pair_f1}, intensity_head LR = {current_intensity_lr:.6f}, above_threshold_count = {self.above_threshold_count}")

            for j in trange(self.trainset.batch_count):
                self.model.train()
                # 获取批次数据
                # sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = self.trainset.get_batch(j) 
                sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes, intensity_tagging_matrices = self.trainset.get_batch(j)
                                                                                                                                    #, intensities , batch_mean, batch_std
                # logits, logits1, sim_matrices = self.model(bert_tokens, masks)
                logits, logits1, sim_matrices, intensity_logits = self.model(bert_tokens, masks)



                # sentiment損失計算
                # print(f"logits shape:{logits.shape}")
                # print(f"tagging_matrices shape:{tagging_matrices.shape}")
                
                logits_flatten = logits.reshape([-1, logits.shape[3]])
                tagging_matrices_flatten = tagging_matrices.reshape([-1])
                # print(f"logits_flatten shape:{logits_flatten.shape}")
                # print(f"tagging_matrices_flatten shape:{tagging_matrices_flatten.shape}")                
                loss0 = self.f_loss(logits_flatten, tagging_matrices_flatten)
                
                # opinion損失計算
                tags1 = tagging_matrices.clone()
                tags1[tags1>0] = 1
                logits1_flatten = logits1.reshape([-1, logits1.shape[3]])
                tags1_flatten = tags1.reshape([-1]).to(self.args.device)
                loss1 = self.f_loss1(logits1_flatten.float(), tags1_flatten)

                # intensity損失計算               
                # 真實值
                # true_intensities = []
                # pred_intensities = []
                # for batch_idx in range(len(intensity_tagging_matrices)):  # 遍歷 batch 16*100*100*2
                #     intensity_tagging_matrix = intensity_tagging_matrices[batch_idx]  # 100*100*2

                #     # 真實值 有點小bug 因為 true matrix 中可能有多個 triples 如果拿max 那可能有的triples沒被訓練到
                #     v = intensity_tagging_matrix[..., 0].max()  # 取出 V (100*100)
                #     a = intensity_tagging_matrix[..., 1].max()
                #     true_intensities.append([v, a])

                #     # 檢查 intensity_pred 是否存在
                #     if 'intensity_pred' not in locals() or intensity_pred is None:
                #         # 如果 intensity_pred 不存在，初始化為隨機值
                #         intensity_pred = torch.empty_like(intensity_logits).uniform_(4, 6)
                #         v_pred = random.uniform(4, 6)  # 生成单个值
                #         a_pred = random.uniform(4, 6)  # 同样生成单个值
                #         # print(f"v_pred: {v_pred},a_pred: {a_pred}")

                #         # print("no predict")
                #         # print("shape:", intensity_pred_matrix[..., 0].shape)
                #         # print("Min:", intensity_pred_matrix[..., 0].min().item())
                #         # print("Max:", intensity_pred_matrix[..., 0].max().item())
                #         # print("Mean:", intensity_pred_matrix[..., 0].mean().item())
                #         # print("Std Dev:", intensity_pred_matrix[..., 0].std().item())

                #     else:
                #     # 獲取預測值
                #         intensity_pred_matrix = intensity_logits[batch_idx]  # 100*100*2
                        

                #         v_pred = intensity_pred_matrix[..., 0].max()  
                #         a_pred = intensity_pred_matrix[..., 1].max()# 取出 V (100*100)
                #         # print("predict:")
                #     # print(f"v_pred: {v_pred},a_pred: {a_pred}")
                #     # print(f"v_pred : {v_pred}, a_pred: {a_pred}")

                #     pred_intensities.append([v_pred, a_pred])
                
                # # 提取 v 和 a
                # true_v = torch.tensor([item[0] for item in true_intensities], dtype=torch.float, device=self.args.device) # 長度為 16 的 float 張量
                # true_a = torch.tensor([item[1] for item in true_intensities], dtype=torch.float, device=self.args.device)
                # # print(f"true_v: {true_v},true_a: {true_a}")
                # pred_v = torch.tensor([item[0] for item in pred_intensities], dtype=torch.float, device=self.args.device) # 長度為 16 的 float 張量
                # pred_a = torch.tensor([item[1] for item in pred_intensities], dtype=torch.float, device=self.args.device)

                # # 使用 MSE 損失計算
                # v_loss = F.mse_loss(pred_v, true_v)
                # a_loss = F.mse_loss(pred_a, true_a)
 

                # print(f"intensity_logits shape:{intensity_logits.shape}") # 16 80 80 2
                # print(f"intensity_tagging_matrices shape:{intensity_tagging_matrices.shape}")
                
                # --- 生成三元組 Mask ---
                mask = self.generate_triplet_mask(tagging_matrices)

                # --- 過濾有效數據 ---
                valid_intensity_logits = intensity_logits[mask]
                valid_intensity_tagging_matrices = intensity_tagging_matrices[mask]

                # --- 計算損失 ---
                # 避免空數據導致計算問題
                if valid_intensity_logits.numel() > 0:
                    intensity_loss = F.mse_loss(valid_intensity_logits, valid_intensity_tagging_matrices)
                else:
                    intensity_loss = torch.tensor(0.0, device=self.args.device)
                
                
                # intensity_logits_flatten = intensity_logits.reshape([-1, intensity_logits.shape[3]])
                # intensity_tagging_matrices_flatten = intensity_tagging_matrices.reshape([-1, intensity_logits.shape[3]])
                # # print(f"intensity_logits_flatten shape:{intensity_logits_flatten.shape}")
                # # print(f"tagging_matricintensity_tagging_matrices_flattenes_flatten shape:{intensity_tagging_matrices_flatten.shape}")                
                # intensity_loss = F.mse_loss(intensity_logits_flatten, intensity_tagging_matrices_flatten)
                

                # 合併損失
                # intensity_loss = (v_loss + a_loss) / 2
                # self.logging(f" Batch {j} intensity Loss: {intensity_loss}")

                # print("調試點11 Valid Predicted Intensities:", intensity_scores[valid_triplets][:5])  # 有效的預測值
                # print("調試點11 Valid True Intensities:", intensities[valid_triplets][:5])  # 有效的標籤值

                # Debug: 確認損失值
                # print(f"Intensity Loss: {intensity_loss.item()}")
                # print(f"Epoch {i}, Batch {j}, Intensity Loss: {intensity_loss.item()}")

                # print("調試點12 Intensity Loss Value:", intensity_loss.item())

                loss_cl = (sim_matrices * cl_masks).mean()
                
                # if self.contrastive:
                #     loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl
                # else:
                #     loss = loss0 + self.beta_1 * loss1
                
                
                # alpha, beta, gamma = 1, 1, 0.002 # 對應 sentiment, opinion, intensity


                # 動態調整 gamma +  
                # 總損失
                if self.contrastive:
                    # loss =  alpha* loss0 + beta * self.beta_1 * loss1 + self.beta_2 * loss_cl + gamma* intensity_loss
                    loss =  loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl + self.gamma * intensity_loss
                else:
                    loss = loss0 + self.beta_1 * loss1 + intensity_loss
                
                
                # loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl
                epoch_sum_loss.append(loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 裁剪梯度

                self.optimizer.step()

                self.writer.add_scalar('train loss_intensity', intensity_loss.item(), i * self.trainset.batch_count + j + 1)

                self.writer.add_scalar('train loss', loss, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss0', loss0, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss1', loss1, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss_intensity', intensity_loss, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('train loss_cl', loss_cl, i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr1', self.optimizer.param_groups[1]['lr'], i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr2', self.optimizer.param_groups[2]['lr'], i*self.trainset.batch_count+j+1)
                self.writer.add_scalar('lr3', self.optimizer.param_groups[3]['lr'], i*self.trainset.batch_count+j+1)
                
            epoch_avg_loss = sum(epoch_sum_loss) / len(epoch_sum_loss)
            # 记录 GPU 内存和梯度范数
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            total_norm = sum(p.grad.data.norm(2).item()**2 for p in self.model.parameters() if p.grad is not None)**0.5

            self.logging(f"Epoch {i}, Batch {j},Sentiment Loss: {loss0.item()}, Opinion Loss: {loss1.item()}, Intensity Loss: {intensity_loss.item()}")
            self.logging('{}\tAvg loss: {:.10f}'.format(str(datetime.datetime.now()), epoch_avg_loss))
            self.logging(f"GPU Info: {torch.cuda.memory_summary(device=torch.device('cuda:0'), abbreviated=True)}")
            self.logging(f"Batch {j}: GPU Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Gradient Norm: {total_norm:.2f}")
            self.logging(f"GPU Temperature: {get_gpu_temperature()}°C")

            for name, param in self.model.intensity_head.named_parameters():
                if param.grad is not None:
                    self.logging(f"{name} - Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}")
                else:
                    self.logging(f"{name} - Grad is None")

            # if joint_f1_test > self.best_joint_f1_test:
            #     bear = 0
            # else:
            #     bear += 1
            
            if joint_f1 > self.best_joint_f1:
                self.best_joint_f1 = joint_f1
                self.best_joint_epoch = i
                if joint_f1_test > self.best_joint_f1_test:

                    # if joint_f1_test > 74.0:
                    
                    # Ensure the directory exists
                    model_dir = os.path.dirname(self.args.model_save_dir)
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    # model_path = self.args.model_save_dir + self.args.data_version + "-" + self.args.dataset + "-" + str(round(joint_f1_test, 4)) + "-" + 'epoch' + str(i) + '.pt'
                    model_path = self.args.model_save_dir + "best_model_ch.pt"
                    torch.save(self.model, model_path)
                    self.logging(f"Model saved at {model_path}")
                    
                    self.best_joint_f1_test = joint_f1_test
                    self.best_joint_epoch_test = i

            if (j + 1) % 5 ==0:
                gc.collect()  # 清理 Python 中的未引用对象
                torch.cuda.empty_cache()  # 清理緩存

            self.writer.add_scalar('dev f1', joint_f1, i+1)
            self.writer.add_scalar('test f1', joint_f1_test, i+1)
            self.writer.add_scalar('dev precision', joint_precision, i+1)
            self.writer.add_scalar('test precision', joint_precision_test, i+1)
            self.writer.add_scalar('dev recall', joint_recall, i+1)
            self.writer.add_scalar('test recall', joint_recall_test, i+1)
            self.writer.add_scalar('best dev f1', self.best_joint_f1, i+1)
            self.writer.add_scalar('best test f1', self.best_joint_f1_test, i+1)

            self.lr_scheduler.step()

            self.logging('best epoch: {}\tbest dev {} f1: {:.5f}'.format(self.best_joint_epoch+1, self.args.task, self.best_joint_f1))
            self.logging('best epoch: {}\tbest test {} f1: {:.5f}'.format(self.best_joint_epoch_test+1, self.args.task, self.best_joint_f1_test))

        # 关闭TensorBoard写入器
        self.writer.close()


