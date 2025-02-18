import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# 读取命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='输入文件的路径')
parser.add_argument('--output', type=str, required=True, help='输出文件路径')
args = parser.parse_args()

# 读取三个文件的数据
data1 = pd.read_csv(f"{args.input}/HAAM/result/progress.txt", sep='\t', skiprows=[1])
data2 = pd.read_csv(f"{args.input}/Hebb/result/progress.txt", sep='\t', skiprows=[1])
data3 = pd.read_csv(f"{args.input}/None/result/progress.txt", sep='\t', skiprows=[1])

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 训练损失
axs[0, 0].plot(data1['DataNum'], data1['TrainLoss'], label='HAAM', color='red')
axs[0, 0].plot(data2['DataNum'], data2['TrainLoss'], label='Hebb', color='blue')
axs[0, 0].plot(data3['DataNum'], data3['TrainLoss'], label='None', color='green')
axs[0, 0].set_title('Train Loss')
axs[0, 0].legend()

# 测试损失
axs[0, 1].plot(data1['DataNum'], data1['TestLoss'], label='HAAM', color='red')
axs[0, 1].plot(data2['DataNum'], data2['TestLoss'], label='Hebb', color='blue')
axs[0, 1].plot(data3['DataNum'], data3['TestLoss'], label='None', color='green')
axs[0, 1].set_title('Test Loss')
axs[0, 1].legend()

# 测试准确率
axs[1, 0].plot(data1['DataNum'], data1['TestAcc'], label='HAAM', color='red')
axs[1, 0].plot(data2['DataNum'], data2['TestAcc'], label='Hebb', color='blue')
axs[1, 0].plot(data3['DataNum'], data3['TestAcc'], label='None', color='green')
axs[1, 0].set_title('Test Accuracy')
axs[1, 0].legend()

# 调整布局
plt.tight_layout()

# 保存输出图像
plt.savefig(args.output)
