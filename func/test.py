# 参数设置
intensity_initial_lr = 5e-4
intensity_lr_max = 5e-3
wait_epochs = 5

# 初始化
above_threshold_count = 0
current_intensity_lr = intensity_initial_lr
epoch = 0

# 模拟连续满足条件的增长
while current_intensity_lr < intensity_lr_max:
    epoch += 1
    above_threshold_count += 1
    if above_threshold_count >= wait_epochs:
        current_intensity_lr = min(intensity_initial_lr * (1 + (above_threshold_count - wait_epochs) * 0.1), intensity_lr_max)

print(f"需要 {epoch} 个 epoch 达到学习率上限 {intensity_lr_max}")

# intensities = [[0.67, 1.23]]
# predicted_intensities = [[2.345, 0.987]]

# i = 0

# # 處理 self.intensities
# int_intensitys = list(map(lambda x: int(round(x * 10)), intensities[i]))
# print(int_intensitys)  # 輸出: [7, 12]

# # 處理 self.predicted_intensities
# pred_int_intensitys = list(map(lambda x: int(round(x * 10)), predicted_intensities[i]))
# print(pred_int_intensitys)  # 輸出: [23, 10]
