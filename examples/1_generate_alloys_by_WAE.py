"""
case study of using WAE to generator alloys
"""
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from deep_learning.VAE.WAE import WAETrainer
from deep_learning.VAE.base import OneDimensionalDataset

if __name__ == '__main__':
    # 一维向量数据
    dataset = pd.read_csv(r"../project_data/1_phase_ml_dataset.csv")
    elements_col = list(dataset.columns[:-1])
    df_element = dataset[elements_col]
    # 计算合金的组分个数
    df_element["N_alloy"] = df_element.astype(bool).sum(axis=1)
    print("原始数据")
    print(df_element.head())
    input_dim = df_element.shape[1]
    data = torch.Tensor(df_element.values)

    # 创建 WAE 训练器实例
    print("WAE training")
    trainer = WAETrainer(OneDimensionalDataset(data), input_dim=input_dim)
    # 训练模型
    trainer.train(epochs=200)
    print("WAE training finished")
    print("WAE generation")
    # 生成样本
    gen = True
    if gen:
        generated_samples_scaled = trainer.generate_samples(num_samples=10000)
        # 确保tensor在CPU上再转换为numpy
        if generated_samples_scaled.is_cuda:
            generated_samples_scaled = generated_samples_scaled.cpu()
        generated_samples_scaled = generated_samples_scaled.numpy()
        
        # 反归一化得到生成样本 (成分 + 元素个数）
        generated_samples = trainer.data.scaler.inverse_transform(generated_samples_scaled)
        df_gen_source = pd.DataFrame(generated_samples_scaled, columns=df_element.columns)
        df_gen = pd.DataFrame(generated_samples, columns=df_element.columns)
        
        # 获取N_alloy值，确保至少为1（避免无效值）
        N_alloy = df_gen["N_alloy"].round().astype(int).clip(lower=1)
        
        # 去掉N_alloy列
        df_gen_source.drop(columns=["N_alloy"], inplace=True)
        
        # 每一行 找到前n大的元素
        for index, row in df_gen_source.iterrows():
            n = N_alloy[index]  # 直接使用，不需要乘以-1
            # 确保n在有效范围内（至少为1，最多为列数）
            n = min(max(1, n), len(row))
            top_n_indices = row.nlargest(n).index
            new_row = [0] * len(row)
            for col in top_n_indices:
                new_row[df_gen_source.columns.get_loc(col)] = row[col]
            df_gen_source.loc[index] = new_row
        
        # 确保列顺序一致
        df_gen_source["N_alloy"] = N_alloy
        df_gen_source = df_gen_source[df_element.columns]
        
        # 归一化
        gen = trainer.data.scaler.inverse_transform(df_gen_source.values)
        df_gen = pd.DataFrame(gen, columns=df_element.columns)
        
        # 归一化所有元素和为100，避免除零错误
        element_sums = df_gen[elements_col].sum(axis=1)
        df_gen = df_gen[elements_col].div(element_sums, axis=0) * 100
        
        # save
        df_gen.to_csv('generated_samples_wae.csv', index=False)