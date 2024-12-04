import torch
import math

# 模擬參數
epochs = 20  # 測試 20 個 epoch
gamma_base = 0.0001  # 初始 gamma 值
alpha = 0.2  # 增長速度參數
threshold = 45  # joint_pair_f1 的臨界值
wait_epochs = 3  # 持續超過或低於臨界值的等待 epoch 數

# 模擬 joint_pair_f1 的值
joint_pair_f1_values = [40, 42, 46, 48, 50, 47, 44, 46, 49, 51, 43, 40, 44, 50, 52, 53, 40, 41, 44, 45]

# 記錄超過或低於閾值的持續 epoch 計數
above_threshold_count = 0
below_threshold_count = 0
gamma = gamma_base

# 模擬訓練過程
for i in range(epochs):
    joint_pair_f1 = joint_pair_f1_values[i]  # 獲取當前 epoch 的 joint_pair_f1

    if joint_pair_f1 > threshold:
        above_threshold_count += 1  # 超過閾值計數器 +1
        below_threshold_count = 0  # 清零低於閾值計數器
    elif joint_pair_f1 < threshold:
        below_threshold_count += 1  # 低於閾值計數器 +1
        above_threshold_count = 0  # 清零超過閾值計數器
    else:
        above_threshold_count = 0
        below_threshold_count = 0

    # 當超過閾值持續達到條件，逐漸增加 gamma
    if above_threshold_count >= wait_epochs:
        gamma = gamma_base * math.exp(alpha * (above_threshold_count - wait_epochs))
    # 當低於閾值持續達到條件，逐漸減少 gamma
    elif below_threshold_count >= wait_epochs:
        gamma = gamma_base / math.exp(alpha * (below_threshold_count - wait_epochs))
    else:
        gamma = gamma_base  # 保持初始值

    print(f"Epoch {i+1}: joint_pair_f1 = {joint_pair_f1}, gamma = {gamma:.6f}, above_count = {above_threshold_count}, below_count = {below_threshold_count}")
