import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn as nn
import torch.nn.utils as nn_utils

from utils.common_utils import stop_words
from tools.evaluate import evaluate

from tqdm import trange
from utils.plot_utils import gather_features, plot_pca, plot_pca_3d
import copy
import os 
import gc
import subprocess
import numpy as np

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

    def train(self):
        bear = 0
        last = self.last
        
        # 定義保存的模型路徑
        saved_model_path = os.path.join(r"/mnt/md0/chen-wei/zi/MiniConGTS_copy_ch_cantrain/modules/mols/saved_models/best_model_ch.pt")

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

            self.logging('\n\nEpoch:{}'.format(i+1))
            self.logging(f"contrastive: {self.contrastive} | bear/max: {bear}/{self.bear_max} | last: {last}")

            epoch_sum_loss = []

            for j in trange(self.trainset.batch_count):
                self.model.train()
                # 获取批次数据
                # sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = self.trainset.get_batch(j)
                sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes, intensities , batch_mean, batch_std= self.trainset.get_batch(j)

                # logits, logits1, sim_matrices = self.model(bert_tokens, masks)
                logits, logits1, sim_matrices, intensity_logits = self.model(bert_tokens, masks)
                
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

                # intensity損失計算                 # 過濾補零部分
                valid_triplets = intensities.sum(dim=2) > 0  # shape: (batch_size, max_triplets)

                intensity_logits = intensity_logits[:, :valid_triplets.size(1), :]
                intensity_logits = (intensity_logits - torch.mean(intensity_logits)) / torch.std(intensity_logits)


                # print("調試點11 Valid Predicted Intensities:", intensity_scores[valid_triplets][:5])  # 有效的預測值
                # print("調試點11 Valid True Intensities:", intensities[valid_triplets][:5])  # 有效的標籤值
                
                # intensity回归损失：将 logits 转换为 scores 进行计算
                # 縮減多餘維度，使用平均池化
                # intensity_scores = intensity_logits.mean(dim=2, keepdim=True)  # 保留維度

                # 確保形狀與標籤一致
                # intensity_scores = intensity_scores.squeeze(dim=2)  # 去掉多餘的維度
                # print(f"Adjusted intensity_logits shape: {intensity_logits.shape}")
                # print(f"intensities shape: {intensities.shape}")
                
                expanded_intensities = intensities.unsqueeze(2)  # 增加一個維度表示 max_length
                expanded_intensities = expanded_intensities.expand(-1, -1, intensity_logits.size(2), -1)  # max_length 從 `intensity_logits` 繼承
                # print(f"Expanded intensities shape: {expanded_intensities.shape}")  # 應該為 [batch_size, max_triplets, max_length, 2]

                
                # Step 4: 將 `valid_triplets` 廣播到 `[batch_size, max_triplets, max_length, 2]`
                valid_triplets_mask = valid_triplets.unsqueeze(2).unsqueeze(3).expand(-1, -1, intensity_logits.size(2), intensity_logits.size(3))

                # Step 5: 過濾無效部分
                valid_intensity_logits = intensity_logits[valid_triplets_mask].view(-1, 2)  # 只保留有效部分
                valid_intensities = expanded_intensities[valid_triplets_mask].view(-1, 2)  # 只保留有效部分

                # Step 6: 使用 MSELoss 計算損失
                intensity_loss = self.intensity_loss_fn(valid_intensity_logits, valid_intensities)
                
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

                # 定义可学习权重
                self.alpha = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
                self.gamma = torch.nn.Parameter(torch.tensor(0.002), requires_grad=True)

                # 将权重加入优化器
                self.optimizer.add_param_group({'params': [self.alpha, self.beta, self.gamma]})
                
                # 總損失
                if self.contrastive:
                    # loss =  alpha* loss0 + beta * self.beta_1 * loss1 + self.beta_2 * loss_cl + gamma* intensity_loss
                    loss =  self.alpha* loss0 + self.beta * self.beta_1 * loss1 + self.beta_2 * loss_cl + self.gamma* intensity_loss
                else:
                    loss = loss0 + self.beta_1 * loss1 + intensity_loss
                
                
                # loss = loss0 + self.beta_1 * loss1 + self.beta_2 * loss_cl
                epoch_sum_loss.append(loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                # nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 裁剪梯度

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
            joint_precision, joint_recall, joint_f1 = self.evaluate(self.model, self.devset, self.stop_words, self.logging, self.args)
            joint_precision_test, joint_recall_test, joint_f1_test = self.evaluate(self.model, self.testset, self.stop_words, self.logging, self.args)



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

            if (j + 1) % 5 == 0 ==0:
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
