"""
PCA projection and visualizations
"""
import pandas as pd
import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class PCAVisualizer(object):
    def __init__(self, n_components=2, n_clusters=3):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters)   # 默认3类
        self.dimension = None
        self.x = None
        self.x_scaled = None
        self.columns = None
    def fit(self, X):
        """
        对数据进行标准化和PCA拟合。

        参数:
        X -- 特征数据，二维numpy数组或DataFrame。
        Y -- 分类标签，一维numpy数组或Series。
        """
        # 标准化特征数据
        X_scaled = self.scaler.fit_transform(X)
        self.x_scaled = X_scaled
        self.dimension = X.shape[1]
        self.x = X
        if isinstance(X, pd.DataFrame):
            self.columns = list(X.columns)
        # 使用PCA降维
        self.pca.fit(X_scaled)
        # 使用KMeans聚类
        self.kmeans.fit(self.pca.transform(X_scaled))


    def visualize_kmeans(self, X, Y, save=False, filename='kmeans_pca_visualization.png'):
        """
        对X进行k-means聚类并在PCA降维后的空间中可视化。
        使用不同的形状表示聚类结果，颜色表示Y的分类标签。
        """
        # 将数据投影到PCA空间
        X_pca = self.transform(X)
        # 进行k-means聚类
        labels = self.kmeans.predict(X_pca)
        kmeans_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(self.n_components)])
        kmeans_df['Cluster'] = labels
        kmeans_df['Label'] = Y

        # 使用seaborn的scatterplot进行可视化
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=kmeans_df, x='PC1', y='PC2', hue='Label', style='Cluster', palette='Set2')
        plt.title('K-Means on PCA Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Label')
        plt.xticks([])
        plt.yticks([])
        # 保存或显示图像
        if save:
            plt.savefig(filename)
        else:
            plt.show()

    def transform(self, X):
        """
        将新数据投影到PCA降维后的空间。

        参数:
        X -- 特征数据，二维numpy数组或DataFrame。

        返回:
        X_pca -- 投影到PCA空间的数据。
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def visualize(self, X, Y, save=False, filename='pca_visualization.png'):
        """
        使用PCA将数据降维到2维，并使用seaborn的scatterplot进行可视化。

        参数:
        X -- 特征数据，二维numpy数组或DataFrame。
        Y -- 分类标签，一维numpy数组或Series。
        save -- 是否保存图像，默认为False。
        filename -- 保存图像的文件名，默认为'pca_visualization.png'。
        """
        # 将数据投影到PCA空间
        X_pca = self.transform(X)

        # 创建DataFrame以便使用seaborn
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(self.n_components)])
        pca_df['Label'] = Y

        # 使用seaborn的scatterplot进行可视化
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Label', style='Label', palette='Set2')
        plt.title('PCA Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Label')

        # 保存或显示图像
        if save:
            plt.savefig(filename)
        else:
            plt.show()

    def save(self, filename):
        """
        将PCAVisualizer对象保存到文件。

        参数:
        filename -- 保存对象的文件名。
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        从文件读取PCAVisualizer对象。
        参数:
        filename -- 包含对象的文件名。
        返回:
        加载的PCAVisualizer对象。
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def inverse_design(self, x, y, n_solutions=2):
        """
        找到n_solutions个高维空间中的点，使得其PCA变换后尽可能接近给定的低维空间中的点。
        参数:
        target_point -- 低维空间的目标点，一维numpy数组。
        n_solutions -- 需要找到的解的个数。
        返回:
        df_design_points -- 高维空间中的点的dataframe;
        """
        target_point = np.array([[x, y]])
        # 定义目标函数
        def objective(high_dim_point, target):
            """distance at low """
            low_dim_point = self.transform(high_dim_point.reshape(1, -1))
            return np.linalg.norm(low_dim_point - target) ** 2
        # 找到距离目标点最近的n个高维样本标准化后的数据
        distances = cdist(self.pca.transform(self.x_scaled), target_point).ravel()
        indices = np.argsort(distances)[:n_solutions]
        # 初始化最优解
        optimized_points = np.zeros((n_solutions, self.dimension))

        for i, idx in enumerate(indices):
            initial_guess = self.x_scaled[idx]  # 初始猜测为某个临近样本点
            # optimize feature space to minimize the difference between the target point and the transformed point
            result = minimize(objective, initial_guess, args=(target_point,), method='BFGS')
            optimized_points[i] = result.x

        for i in range(n_solutions):
            point_x = self.scaler.inverse_transform(optimized_points[i].reshape(1, -1))
            print("point", point_x)
            print("transformed point", self.pca.transform(point_x))
            print("error", objective(optimized_points[i], target_point))
        df_design_points = pd.DataFrame(self.scaler.inverse_transform(optimized_points), columns=self.columns)
        return df_design_points
