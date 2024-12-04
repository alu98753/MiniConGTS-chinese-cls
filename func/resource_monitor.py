import psutil
import time
import csv
import os

# 指定目录和文件路径
log_dir = '/mnt/md0/chen-wei/zi/MiniConGTS_copy/func'
log_file = os.path.join(log_dir, 'resource_log.csv')

# 确保目录存在，如果不存在则创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 初始化 CSV 文件（如果文件不存在，则写入表头）
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'CPU (%)', 'Memory (%)'])  # 表头

# 开始记录资源状态
while True:
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # 写入当前时间、CPU 使用率和内存使用率
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), psutil.cpu_percent(), psutil.virtual_memory().percent])
    time.sleep(60)  # 每分钟记录一次
