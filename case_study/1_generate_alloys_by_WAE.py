"""
case study of using WAE to generator alloys
"""
import pandas as pd
import torch
from util.deep_learning.VAE.WAE import WAETrainer
from util.deep_learning.VAE.base import OneDimensionalDataset

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
        generated_samples_scaled = trainer.random_generate_samples(num_samples=10000).numpy()
        # 反归一化得到生成样本 (成分 + 元素个数）
        generated_samples = trainer.data.scaler.inverse_transform(generated_samples_scaled)
        df_gen_source = pd.DataFrame(generated_samples_scaled, columns=df_element.columns)
        # 去掉N_alloy列
        df_gen_source.drop(columns=["N_alloy"], inplace=True)
        df_gen = pd.DataFrame(generated_samples, columns=df_element.columns)
        N_alloy = -1 * df_gen["N_alloy"].round().astype(int).copy()
        # print(N_alloy.head())
        # print(df_gen.head())
        # print(df_gen_source.head())
        # 每一行 找到前n大的元素
        for index, row in df_gen_source.iterrows():
            n = N_alloy[index] * -1  # 找前几个最大元素
            top_n_indices = row.nlargest(n).index
            new_row = [0] * len(row)
            for col in top_n_indices:
                new_row[df_gen_source.columns.get_loc(col)] = row[col]
            df_gen_source.loc[index] = new_row
        df_gen_source["N_alloy"] = N_alloy
        # 归一化
        gen = trainer.data.scaler.inverse_transform(df_gen_source.values)
        df_gen = pd.DataFrame(gen, columns=df_element.columns)
        # 归一化所有元素和为100
        df_gen = df_gen[elements_col].div(df_gen[elements_col].sum(axis=1), axis=0) * 100
        # save
        df_gen.to_csv('generated_samples_wae.csv', index=False)