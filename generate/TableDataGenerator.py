import pandas as pd
import torch
from util.deep_learning.VAE.WAE import WAETrainer
from util.deep_learning.VAE.base import OneDimensionalDataset


class TableDataGenerator:
    def __init__(self, df, latent_dims=None):
        if latent_dims is None:
            latent_dims = [1, 5, 10, 20]
        self.trainer = None
        # 分离数值列和字符串列
        self.numeric_cols = df.select_dtypes(include=['number']).columns
        self.string_cols = df.select_dtypes(include=['object']).columns
        # 对字符串列进行 one - hot 编码
        df_encoded = pd.get_dummies(df, columns=self.string_cols)
        self.input_dim = df_encoded.shape[1]  # 列数量
        self.data = torch.Tensor(df_encoded.values)
        self.column = df_encoded.columns
        self.original_df = df
        self.latent_dims = latent_dims

    def train(self):
        self.trainer = WAETrainer(OneDimensionalDataset(self.data), input_dim=self.input_dim,
                                  latent_dims=self.latent_dims)
        self.trainer.train(epochs=200)

    def generate(self, num_samples=100):
        generated_samples_scaled = self.trainer.generate_samples(num_samples=num_samples).numpy()
        generated_samples = self.trainer.data.scaler.inverse_transform(generated_samples_scaled)
        df_gen_encoded = pd.DataFrame(generated_samples, columns=self.column)
        # 逆变换 one - hot 编码
        df_gen = df_gen_encoded.copy()
        for col in self.string_cols:
            original_categories = self.original_df[col].unique()
            encoded_cols = [col + '_' + str(cat) for cat in original_categories]
            df_gen[col] = df_gen_encoded[encoded_cols].idxmax(axis=1).str.replace(col + '_', '')
            df_gen = df_gen.drop(encoded_cols, axis=1)
        return df_gen
