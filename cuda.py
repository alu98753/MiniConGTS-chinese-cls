import torch
import variable

# # 創建與 x 形狀相同的空張量，並指定資料型別為浮點型
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# empty_tensor = torch.empty_like(x, dtype=torch.float32)
# print(empty_tensor)  # 正確初始化後的結果

# # 使用正態分佈初始化，並限制值範圍
# empty_tensor.data.normal_(0, 1).clamp_(-0.5, 0.5)

# print(empty_tensor)  # 正確初始化後的結果
del variable
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=torch.device('cuda:0'), abbreviated=True))

