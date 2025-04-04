"""
特征解释器
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from typing import Dict, Any, List, Tuple, Optional
import unittest


class FeatureExplainer(object):
    def __init__(self, model: Any, X: pd.DataFrame, y: pd.Series):
        """
        参数:
        model: 训练好的模型
        X: 特征矩阵
        y: 目标变量
        """
        self.model = model
        self.X = X
        self.y = y
        self.perm_importance = None
        self.perm_importance_data = None

    def calculate_permutation_importance(self,
                                         n_repeats: int = 100,
                                         random_state: int = 42) -> dict[str, Any]:
        """
        计算特征置换重要性
        
        参数:
        n_repeats: 置换重复次数
        random_state: 随机种子
        
        返回:
        pd.DataFrame: 特征重要性数据
        """
        # 计算置换重要性
        result = permutation_importance(
            self.model, self.X, self.y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        # 整理结果
        importance_data = {
            'feature': list(self.X.columns),
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
            'importance_values': result.importances.T.tolist(),
        }

        # 按重要性排序

        self.perm_importance = result
        self.perm_importance_data = importance_data

        return importance_data

    def plot_permutation_importance(self,
                                    top_n: Optional[int] = None,
                                    fig_size: Tuple[float, float] = (10, 6),
                                    title: str = "特征置换重要性") -> Tuple[plt.Figure, plt.Axes]:

        if self.perm_importance_data is None:
            raise ValueError("请先运行 calculate_permutation_importance 方法")

        # 准备数据
        plot_data = self.perm_importance_data
        if top_n is not None:
            plot_data = plot_data.head(top_n)

        # 创建图形
        fig, ax = plt.subplots(figsize=fig_size)

        # 绘制条形图
        bars = ax.bar(
            range(len(plot_data)),
            plot_data['importance_mean'],
            yerr=plot_data['importance_std'],
            capsize=5
        )

        # 设置图形属性
        ax.set_title(title)
        ax.set_xlabel("特征")
        ax.set_ylabel("重要性得分")
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data['feature'], rotation=45, ha='right')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()

        return fig, ax

    def get_permutation_importance_data(self) -> Optional[pd.DataFrame]:
        """
        获取特征置换重要性数据
        
        返回:
        Optional[pd.DataFrame]: 特征重要性数据，如果未计算则返回None
        """
        return self.perm_importance_data

    def get_permutation_importance_plot_data(self,
                                             top_n: Optional[int] = None) -> Dict[str, Any]:
        """
        获取特征置换重要性绘图数据
        
        参数:
        top_n: 显示前N个特征，None表示显示所有
        
        返回:
        Dict[str, Any]: 绘图数据
        """
        if self.perm_importance_data is None:
            return None

        plot_data = self.perm_importance_data
        if top_n is not None:
            plot_data = plot_data.head(top_n)

        return {
            'features': plot_data['feature'].tolist(),
            'importance_mean': plot_data['importance_mean'].tolist(),
            'importance_std': plot_data['importance_std'].tolist(),
            'p_values': plot_data['p_values'].tolist()
        }


class TestFeatureExplainer(unittest.TestCase):
    def setUp(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        self.y = pd.Series([0, 1, 0, 1, 0])
        self.model.fit(self.X, self.y)
        self.explainer = FeatureExplainer(self.model, self.X, self.y)

    def test_calculate_permutation_importance(self):
        result = self.explainer.calculate_permutation_importance()
        print(result)

    def test_plot_permutation_importance_with_top_n(self):
        fig, ax = self.explainer.plot_permutation_importance(top_n=1)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_get_permutation_importance_plot_data_with_top_n(self):
        data = self.explainer.get_permutation_importance_plot_data(top_n=1)
        self.assertIsNotNone(data)

    def test_calculate_permutation_importance_with_top_n(self):
        data = self.explainer.calculate_permutation_importance(top_n=1)
        self.assertIsNotNone(data)
