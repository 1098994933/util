"""
1 dimension VAE for alloy generation
输入和输出的范围标准化到 0~1
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class VAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class OneDimensionalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.min_val = torch.min(self.data)
        self.max_val = torch.max(self.data)
        self.data_normalized = (self.data - self.min_val) / (self.max_val - self.min_val)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data_normalized[index]

class VAETrainer:
    """
    自动划分20% 数据 作为验证集
    """
    def __init__(self, data, input_dim, latent_dim, batch_size=128, learning_rate=0.0001):
        self.data = data
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 划分训练集和验证集
        n = len(data)
        train_size = int(0.8 * n)
        val_size = n - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(data, [train_size, val_size])

        self.model = VAEModel(input_dim, latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.best_val_loss = float('inf')
        self.patience = 10
        self.epochs_without_improvement = 0

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD) / x.size(0)

    def train(self, epochs):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch)
                loss = self.loss_function(recon_batch, batch, mu, logvar)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_data)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    recon_batch, mu, logvar = self.model(batch)
                    loss = self.loss_function(recon_batch, batch, mu, logvar)
                    val_loss += loss.item()

            val_loss /= len(self.val_data)

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # 保存模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.epochs_without_improvement += 1
            # early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    def generate_samples(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.model.decode(z)
            # 逆标准化
            samples_denormalized = samples * (self.data.max_val - self.data.min_val) + self.data.min_val
        return samples_denormalized.numpy()

if __name__ == '__main__':
    # 模拟一维向量数据
    dataset = pd.read_csv(r"D:\PycharmProjects\0_高熵合金\data\1_phase_ml_dataset.csv")
    elements_col = list(dataset.columns[:-1])
    df_element = dataset[elements_col]
    # 计算合金的组分个数
    df_element["N_alloy"] = df_element.astype(bool).sum(axis=1)
    print(df_element.head())
    input_dim = df_element.shape[1]

    data = torch.Tensor(df_element.values)

    #data = torch.randn(1000, 50)

    # 创建 VAE 训练器实例
    vae_trainer = VAETrainer(OneDimensionalDataset(data), input_dim=input_dim, latent_dim=10)

    # 训练模型
    vae_trainer.train(epochs=200)

    # 生成样本
    gen = True
    if gen:
        generated_samples = vae_trainer.generate_samples(num_samples=100)
        print(generated_samples)
        df_gen = pd.DataFrame(generated_samples,columns=df_element.columns)
        df_gen["N_alloy"] = -1 * df_gen["N_alloy"].round().astype(int)
        print(df_gen)
        for index, row in df_gen.iterrows():
            n = int(row['N_alloy']) * -1
            top_n_indices = row.nlargest(n).index
            new_row = [0] * len(row)
            new_row[-1] = n
            for col in top_n_indices:
                new_row[df_gen.columns.get_loc(col)] = row[col]
            df_gen.loc[index] = new_row
        # 归一化
        df_gen = df_gen[elements_col].div(df_gen[elements_col].sum(axis=1), axis=0) * 100
        print(df_gen)
        df_gen.to_csv('generated_samples.csv', index=False)