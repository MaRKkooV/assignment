import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 读取数据（拼接训练集和测试集）
df_train = pd.read_csv("train_data.csv")
df = df_train

# 删除无效列
df = df.drop(columns=["No"], errors="ignore")
df = df.dropna()

# 风向编码 (对 wnd_dir 列进行编码)
df['wnd_dir'] = LabelEncoder().fit_transform(df['wnd_dir'])

# 提取特征（修改为新的列名）
features = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data = df[features].values

# 归一化
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 构造时间序列样本
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])  # 目标为污染值 ('pollution')
    return np.array(X), np.array(y)

seq_len = 24
X, y = create_sequences(data, seq_len)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 构建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        # LSTM层，输入维度=input_size，隐藏层单元数=hidden_size，堆叠层数=num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 线性层用于输出预测值
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM输出: out是所有时间步的隐藏状态，_是最终的隐藏状态和细胞状态
        out, _ = self.lstm(x)
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        # 全连接层输出预测结果
        out = self.fc(out)
        return out

# 初始化模型
model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=2)

# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练过程
train_losses = []
test_losses = []

for epoch in range(10):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 每轮测试集评估
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_loss = criterion(test_output, y_test_tensor).item()
        test_losses.append(test_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "lstm_model.pth")

# 绘制训练与测试误差曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train MSE")
plt.plot(test_losses, label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Testing MSE over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# 预测前 100 个样本并绘制预测值与真实值对比图
model.eval()
with torch.no_grad():
    pred_values = model(X_test_tensor).squeeze().numpy()
    real_values = y_test_tensor.squeeze().numpy()

# 绘制预测值与真实值的前 100 个样本对比图
plt.figure(figsize=(10, 5))
plt.plot(real_values[:100], label='Real Pollution')
plt.plot(pred_values[:100], label='Predicted Pollution')
plt.xlabel("Sample Index")
plt.ylabel("Pollution Level")
plt.title("Comparison of Real vs Predicted Pollution (First 100 Samples)")
plt.legend()
plt.grid(True)
plt.show()
