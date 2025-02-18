import matplotlib.pyplot as plt
import pandas as pd
import argparse
# 读取数据

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='输入文件路径')
parser.add_argument('--output', type=str, required=True, help='输出文件路径')
args = parser.parse_args()

input = pd.read_csv(args.input, sep='\t', skiprows=[1])

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# CIFAR 训练损失
axs[0, 0].plot(input['DataNum'], input['TrainLoss'], label='STDP', color='purple')
axs[0,0].set_title('Train Loss')
axs[0, 1].plot(input['DataNum'], input['TestLoss'], label='STDP', color='green')
axs[0,1].set_title('Test Loss')
axs[1, 0].plot(input['DataNum'], input['TestAcc'], label='STDP', color='red')
axs[1,0].set_title('Test Accuracy')

plt.savefig(args.output)