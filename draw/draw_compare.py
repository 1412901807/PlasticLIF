import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# 读取命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--input1', type=str, required=True, help='输入文件1的路径')
parser.add_argument('--input2', type=str, required=True, help='输入文件2的路径')
parser.add_argument('--output', type=str, required=True, help='输出文件路径')
args = parser.parse_args()

# 读取两个文件的数据
data1 = pd.read_csv(args.input1, sep='\t', skiprows=[1])
data2 = pd.read_csv(args.input2, sep='\t', skiprows=[1])

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# CIFAR 训练损失
axs[0, 0].plot(data1['DataNum'], data1['TrainLoss'], label='File1 - Train Loss', color='purple')
axs[0, 0].plot(data2['DataNum'], data2['TrainLoss'], label='File2 - Train Loss', color='blue')
axs[0, 0].set_title('Train Loss')
axs[0, 0].legend()

# CIFAR 测试损失
axs[0, 1].plot(data1['DataNum'], data1['TestLoss'], label='File1 - Test Loss', color='purple')
axs[0, 1].plot(data2['DataNum'], data2['TestLoss'], label='File2 - Test Loss', color='blue')
axs[0, 1].set_title('Test Loss')
axs[0, 1].legend()

# CIFAR 测试准确率
axs[1, 0].plot(data1['DataNum'], data1['TestAcc'], label='File1 - Test Accuracy', color='purple')
axs[1, 0].plot(data2['DataNum'], data2['TestAcc'], label='File2 - Test Accuracy', color='blue')
axs[1, 0].set_title('Test Accuracy')
axs[1, 0].legend()

# 调整布局
plt.tight_layout()

# 保存输出图像
plt.savefig(args.output)

# 输出差值最大
# 计算差值
train_loss_diff = np.abs(data1['TrainLoss'] - data2['TrainLoss'])
test_loss_diff = np.abs(data1['TestLoss'] - data2['TestLoss'])
test_acc_diff = np.abs(data1['TestAcc'] - data2['TestAcc'])

# 找到差值最大的x值
max_train_loss_idx = train_loss_diff.idxmax()
max_test_loss_idx = test_loss_diff.idxmax()
max_test_acc_idx = test_acc_diff.idxmax()

print(f"最大训练损失差值发生在 x = {data1['DataNum'][max_train_loss_idx]}, 差值 = {train_loss_diff[max_train_loss_idx]}")
print(f"最大测试损失差值发生在 x = {data1['DataNum'][max_test_loss_idx]}, 差值 = {test_loss_diff[max_test_loss_idx]}")
print(f"最大测试准确率差值发生在 x = {data1['DataNum'][max_test_acc_idx]}, 差值 = {test_acc_diff[max_test_acc_idx]}")