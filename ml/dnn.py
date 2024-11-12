import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import logging


class RegressionDNN(nn.Module):
    """
    DNN with 3 fully connect layers for regression
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(RegressionDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # no
        return x


# region dataloader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class DatasetFromDataFrame(Dataset):
    def __init__(self, df, feature_col, targets_col):
        """
        df: pd.DataFrame
        """
        self.features = df[feature_col].apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.target = df[targets_col].apply(pd.to_numeric, errors='coerce').fillna(0).values

        # # 标准化特征（可选）
        # self.scaler = StandardScaler()
        # self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # return features and targets are torch.Tensor
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.target[idx], dtype=torch.float32)
        )



def calculate_r2(model, data_loader):
    """
    计算模型的R²分数。
    :param model: 训练好的模型。
    :param data_loader: 包含测试数据的DataLoader。
    :return: R²分数。
    """
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    total_count = 0

    with torch.no_grad():  # 在评估模式下，不计算梯度
        for batch_data in data_loader:
            inputs, targets = batch_data
            outputs = model(inputs)
            targets = targets.view(-1)  # 确保目标是一维的
            outputs = outputs.view(-1)  # 确保预测也是一维的

    # 计算R²分数
    r2 = sklearn.metrics.r2_score(outputs, targets)
    return r2

def train_dnn_by_dataset(df, model,features_col=None, targets_col=None):
    """
    train DNN model by dataset to predict targets from alloy composition to targets
    """
    print(f'Start Training DNN with input size is {model.input_dim}, output_dim is {model.output_dim}')
    n_of_targets = model.output_dim
    if features_col is None:
        features_col = list(df.columns[:-n_of_targets])
    if targets_col is None:
        targets_col = df.columns[-n_of_targets:]
    print('features_col', features_col)
    print('targets_col', targets_col)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # create DataLoader

    train_dataset = DatasetFromDataFrame(train_df, features_col, targets_col)
    test_dataset = DatasetFromDataFrame(test_df, features_col, targets_col)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 你可以选择其他的优化器和学习率

    # training
    num_epochs = 500
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            inputs, targets = batch_data
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # update parameters
        # 每20个epoch 评估一次模型
        if epoch % 20 == 0:
            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                r2_score = calculate_r2(model, test_loader)
                print(f"Epoch {epoch}: test R² Score: {r2_score}")
            model.train()  # set model to train mode
    # save model
    torch.save(model, './model/final_model.pth')
    print('Final model saved')

def train_dnn(df):
    """
    df = features + target
    """
    # 自定义输入维度
    element_col = list(df.columns)[:-1]
    # 分割数据集为训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # 定义特征列和目标列
    features = element_col
    target = '熔点'

    input_dimension = len(features)
    output_dimension = 1

    # 创建训练集和测试集的Dataset和DataLoader
    train_dataset = DatasetFromDataFrame(train_df, features, target)
    test_dataset = DatasetFromDataFrame(test_df, features, target)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # endregion
    # 创建模型实例
    model = RegressionDNN(input_dim=input_dimension, output_dim=output_dimension)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 你可以选择其他的优化器和学习率

    # training
    num_epochs = 500
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            inputs, targets = batch_data
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        # 每20个epoch评估一次模型
        if epoch % 20 == 0:
            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                r2_score = calculate_r2(model, test_loader)
                print(f"Epoch {epoch}: R² Score: {r2_score}")
            model.train()  # 继续训练模式

    # save model
    torch.save(model, './model/final_model.pth')
    print('Final model saved')

if __name__ == '__main__':
    # 自定义输入维度
    dataset_file = r'./data/dataset.xlsx'
    sheet_name = "Sheet1"
    Y_col = '熔点'
    df = pd.read_excel(dataset_file, sheet_name=sheet_name)
    element_col = list(df.columns)[1:-1]
    # 分割数据集为训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # 定义特征列和目标列
    features = element_col
    target = '熔点'
    logging.info("features", features)
    logging.info("Y_col", target)

    input_dimension = len(features)  # 例如，你可以根据需要更改这个值
    output_dimension = 1  # 对于回归任务，通常输出维度为1
    # 创建训练集和测试集的Dataset和DataLoader
    train_dataset = DatasetFromDataFrame(train_df, features, target)
    test_dataset = DatasetFromDataFrame(test_df, features, target)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # endregion
    # 创建模型实例
    model = RegressionDNN(input_dim=input_dimension, output_dim=output_dimension)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 你可以选择其他的优化器和学习率

    # training
    num_epochs = 500
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            inputs, targets = batch_data
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        # 每20个epoch评估一次模型
        if epoch % 50 == 0:
            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                r2_score = calculate_r2(model, test_loader)
                print(f"Epoch {epoch}: R² Score: {r2_score}")
            model.train()  # 继续训练模式

    # save model
    torch.save(model, './model/final_model.pth')
    print('Final model saved')
