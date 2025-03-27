import pandas as pd
import torch
from deep_learning.VAE.WAE import WAETrainer
from deep_learning.VAE.base import OneDimensionalDataset
from sklearn.preprocessing import LabelEncoder


class TableDataGenerator:
    def __init__(self, df, latent_dims=None):
        if latent_dims is None:
            latent_dims = [1, 5, 10, 20]
        self.trainer = None
        self.numeric_cols = df.select_dtypes(include=['number']).columns
        self.string_cols = df.select_dtypes(include=['object']).columns
        self.label_encoders = {}
        self.original_df = df
        self.latent_dims = latent_dims
        self.encoded_data = df.copy()

        # 对字符串列使用 LabelEncoder 编码
        for col in self.string_cols:
            le = LabelEncoder()
            self.encoded_data[col] = le.fit_transform(self.encoded_data[col])
            self.label_encoders[col] = le

        self.data = torch.Tensor(self.encoded_data.values)
        self.input_dim = self.data.shape[1]

    def train(self, epochs=200):
        self.trainer = WAETrainer(OneDimensionalDataset(self.data), input_dim=self.input_dim,
                                  latent_dims=self.latent_dims)
        self.trainer.train(epochs=epochs)

    def generate(self, num_samples=100):
        generated_samples_scaled = self.trainer.generate_samples(num_samples=num_samples).numpy()
        generated_samples = self.trainer.data.scaler.inverse_transform(generated_samples_scaled)
        df_gen_encoded = pd.DataFrame(generated_samples, columns=self.encoded_data.columns)
        df_gen = df_gen_encoded.copy()

        # 对字符串列使用 LabelEncoder 解码
        for col in self.string_cols:
            le = self.label_encoders[col]
            df_gen[col] = le.inverse_transform(df_gen[col].astype(int))

        return df_gen
