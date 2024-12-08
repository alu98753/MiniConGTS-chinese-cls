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
    def __init__(self, model, trainset, devset, testset, optimizer, criterion, lr_scheduler , lr_scheduler_valence,lr_scheduler_arousal , args, logging, beta_1, beta_2, bear_max, last, plot=False):
        self.model = model
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.optimizer = optimizer
        # self.criterion = criterion
        self.f_loss = criterion[0]  # sentiment損失
        self.f_loss1 = criterion[1]  # opinion損失
        self.f_loss2 = criterion[2]  # v
        self.f_loss3 = criterion[3]  # a
        
        # self.intensity_loss_fn = nn.MSELoss()  # intensity損失

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_valence = lr_scheduler_valence
        self.lr_scheduler_arousal = lr_scheduler_arousal
        self.best_joint_f1 = 0
        self.best_joint_f1_test = 0
        self.best_joint_epoch = 0
        self.best_joint_epoch_test = 0
        self.joint_intensity_f1_test = 0
        self.best_joint_intensity_f1 = 0
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
      
        self.gamma = 0.3  # gamma 初始值
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

    # 動態計算損失權重
    def compute_dynamic_weights(self, losses ,priority_indices ):
        if type(losses) != torch.Tensor:
            losses = torch.tensor(losses)
        
        weights = torch.div(losses , torch.sum(losses)) * losses.shape[0]

        return weights[3],weights[4]

        
    def train(self):
        bear = 0
        last = self.last
        w2 = w3 =1
        
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
            epoch_valence_loss = []
            epoch_arousal_loss = []
            joint_precision, joint_recall, joint_f1 ,pair_results ,joint_intensity_f1= self.evaluate(self.model, self.devset, self.stop_words, self.logging, self.args)
            joint_precision_test, joint_recall_test, joint_f1_test,pair_results_test ,joint_intensity_f1_test = self.evaluate(self.model, self.testset, self.stop_words, self.logging, self.args)

            if pair_results_test > 50:
                self.gamma = 0.5
 
            self.logging('\n\nEpoch:{}'.format(i+1))
            self.logging(f"contrastive: {self.contrastive} | bear/max: {bear}/{self.bear_max} | last: {last}")

            for j in trange(self.trainset.batch_count):
                self.model.train().to(self.args.device)
                # 获取批次数据
                # sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = self.trainset.get_batch(j) 
                sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes,  valence_matrices, arousal_matrices = self.trainset.get_batch(j)
                                                                                                                                    #, intensities , batch_mean, batch_std
                # logits, logits1, sim_matrices = self.model(bert_tokens, masks)
                logits, logits1, sim_matrices, logits_valence, logits_arousal = self.model(bert_tokens, masks)

                # sentiment損失計算
                logits_flatten = logits.reshape([-1, logits.shape[3]])
                tagging_matrices_flatten = tagging_matrices.reshape([-1])
                loss0 = self.f_loss(logits_flatten, tagging_matrices_flatten)
                
                # opinion損失計算
                tags1 = tagging_matrices.clone()
                tags1[tags1>0] = 1
                logits1_flatten = logits1.reshape([-1, logits1.shape[3]])
                tags1_flatten = tags1.reshape([-1]).to(self.args.device)
                loss1 = self.f_loss1(logits1_flatten.float(), tags1_flatten)

                # Valence 損失計算
                logits_valence_flatten = logits_valence.reshape([-1, logits_valence.shape[3]])
                valence_labels_flatten = valence_matrices.reshape([-1])

                # Arousal 損失計算
                logits_arousal_flatten = logits_arousal.reshape([-1, logits_arousal.shape[3]])
                arousal_labels_flatten = arousal_matrices.reshape([-1])
                
                loss2 = self.f_loss2(logits_valence_flatten, valence_labels_flatten)

                loss3 = self.f_loss3(logits_arousal_flatten, arousal_labels_flatten)

                loss_cl = (sim_matrices * cl_masks).mean()
                
                # if self.contrastive:
                #     loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl
                # else:
                #     loss = loss0 + self.beta_1 * loss1
                # if joint_pair_f1 > 45:
                #     losses = [loss0 ,loss1 , loss_cl ,loss2, loss3]
                #     priority_indices = [3, 4]  # loss2 和 loss3 的索引

                #     # 動態調整損失權重
                #     w2, w3 = self.compute_dynamic_weights(losses, priority_indices)
                
                # 合併損失
                if self.contrastive:
                    loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl + self.gamma * ( loss2 +  loss3)
                else:
                    loss = loss0 + self.beta_1 * loss1 + ( loss2 +   loss3) 
                # if self.contrastive:
                #     loss =  self.beta_1 * loss1 + self.beta_2 * loss_cl + self.gamma * ( loss2 +  loss3)
                # else:
                #     loss =  self.beta_1 * loss1 + ( loss2 +   loss3)           
                # 收集損失值

                # loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl
                epoch_sum_loss.append(loss)
                epoch_valence_loss.append(loss2.item())
                epoch_arousal_loss.append(loss3.item())
                self.optimizer.zero_grad()
                loss.backward()
                torch.cuda.synchronize()

                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 裁剪梯度

                self.optimizer.step()
                # 添加学习率打印
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.logging(f"Param group {i}: learning rate = {param_group['lr']}")
                # self.writer.add_scalar('train loss', loss, i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('train loss0', loss0, i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('train loss1', loss1, i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('train valence loss', loss2.item(), i * self.trainset.batch_count + j + 1)
                # self.writer.add_scalar('train arousal loss', loss3.item(), i * self.trainset.batch_count + j + 1)
                # self.writer.add_scalar('train loss_cl', loss_cl, i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('lr1', self.optimizer.param_groups[1]['lr'], i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('lr2', self.optimizer.param_groups[2]['lr'], i*self.trainset.batch_count+j+1)
                # self.writer.add_scalar('lr3', self.optimizer.param_groups[3]['lr'], i*self.trainset.batch_count+j+1)
                
            epoch_avg_loss = sum(epoch_sum_loss) / len(epoch_sum_loss)
            # 計算平均損失
            avg_valence_loss = sum(epoch_valence_loss) / len(epoch_valence_loss)
            avg_arousal_loss = sum(epoch_arousal_loss) / len(epoch_arousal_loss)
            # 记录 GPU 内存和梯度范数
            # allocated = torch.cuda.memory_allocated() / 1024 ** 2
            # reserved = torch.cuda.memory_reserved() / 1024 ** 2
            # total_norm = sum(p.grad.data.norm(2).item()**2 for p in self.model.parameters() if p.grad is not None)**0.5

            self.logging(f"Epoch {i}, Batch {j},Sentiment Loss: {loss0.item()}, Opinion Loss: {loss1.item()}, loss2 (v): {loss2.item()}, loss3 (a): {loss3.item()}")
            self.logging(f"Epoch {i}, Avg Loss: {epoch_avg_loss:.6f}, Valence Loss: {avg_valence_loss:.6f}, Arousal Loss: {avg_arousal_loss:.6f}")
            # self.logging(f"GPU Info: {torch.cuda.memory_summary(device=torch.device('cuda:0'), abbreviated=True)}")
            # self.logging(f"Batch {j}: GPU Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Gradient Norm: {total_norm:.2f}")
            self.logging(f"GPU Temperature: {get_gpu_temperature()}°C")

            for name, param in self.model.cls_linear_valence.named_parameters():
                if param.grad is not None:
                    self.logging(f"{name} - Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}")
                else:
                    self.logging(f"{name} - Grad is None")
            
            for name, param in self.model.cls_linear_arousal.named_parameters():
                if param.grad is not None:
                    self.logging(f"{name} - Grad Min: {param.grad.min()}, Grad Max: {param.grad.max()}")
                else:
                    self.logging(f"{name} - Grad is None")
            # if joint_f1_test > self.best_joint_f1_test:
            #     bear = 0
            # else:
            #     bear += 1
            
            # if joint_intensity_f1 > self.best_joint_intensity_f1:
            #     self.best_joint_intensity_f1 = joint_intensity_f1
            #     self.best_joint_epoch = i
                
            if joint_intensity_f1_test > self.joint_intensity_f1_test:

                if joint_intensity_f1_test > 20.0:
                
                    # Ensure the directory exists
                    model_dir = os.path.dirname(self.args.model_save_dir)
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    model_path = self.args.model_save_dir + 'epoch' + str(i)+ "-" + str(round(joint_intensity_f1_test, 4)) + "-" + '.pt'
                    # model_path = self.args.model_save_dir + "best_model_ch.pt"
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': i,
                        'triplet_intensity_f1_test': joint_intensity_f1_test
                    }, model_path)
                    self.logging(f"Model saved at {model_path}")
                    
                    self.joint_intensity_f1_test = joint_intensity_f1_test
                    self.best_joint_epoch_test = i

            if (j + 1) % 5 ==0:
                gc.collect()  # 清理 Python 中的未引用对象
                torch.cuda.empty_cache()  # 清理緩存

            # self.writer.add_scalar('dev f1', joint_f1, i+1)
            # self.writer.add_scalar('test f1', joint_f1_test, i+1)
            # self.writer.add_scalar('dev precision', joint_precision, i+1)
            # self.writer.add_scalar('test precision', joint_precision_test, i+1)
            # self.writer.add_scalar('dev recall', joint_recall, i+1)
            # self.writer.add_scalar('test recall', joint_recall_test, i+1)
            # self.writer.add_scalar('best dev f1', self.best_joint_f1, i+1)
            # self.writer.add_scalar('best test f1', self.best_joint_f1_test, i+1)
            # 記錄學習率和損失
            self.lr_scheduler.step()
            self.lr_scheduler_valence.step(avg_valence_loss)  # 對 valence 使用 ReduceLROnPlateau
            self.lr_scheduler_arousal.step(avg_arousal_loss)  # 對 arousal 使用 ReduceLROnPlateau
            # self.logging('best epoch: {}\tbest dev {} f1: {:.5f}'.format(self.best_joint_epoch+1, self.args.task, self.best_joint_f1))
            self.logging('best epoch: {}\tbest test {} f1: {:.5f}'.format(self.best_joint_epoch_test+1, self.args.task, self.joint_intensity_f1_test))

        # 关闭TensorBoard写入器
        self.writer.close()


