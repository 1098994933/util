"""
pca
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def pca_visualization(X, Y, save=False, filename='pca_visualization.png'):
    """
    使用PCA将数据降维到2维，并使用seaborn的scatterplot进行可视化。

    参数:
    X -- 特征数据，二维numpy数组或DataFrame。
    Y -- 分类标签，一维numpy数组或Series。
    save -- 是否保存图像，默认为False。
    filename -- 保存图像的文件名，默认为'pca_visualization.png'。
    """
    # 将X和Y转换为numpy数组，以确保兼容性
    X = np.array(X)
    Y = np.array(Y)

    # 标准化特征数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 创建DataFrame以便使用seaborn
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Label'] = Y

    # 使用seaborn的scatterplot进行可视化
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Label', palette='viridis')
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label')

    # 保存或显示图像
    if save:
        plt.savefig(filename)
    else:
        plt.show()

# 示例使用
# 假设X和Y是你的数据和标签
# pca_visualization(X, Y, save=True)

