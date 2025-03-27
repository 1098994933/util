import unittest
import pandas as pd
import numpy as np
from math import ceil


class DataFrameGenerator(object):
    def __init__(self, x: pd.DataFrame):
        """
        初始化生成器 x是一个全numerical的 dataframe
        
        参数:
        X: pd.DataFrame - 原始特征数据框
        """
        self.x = x
        self.features = list(x.columns)
        # 存储每个特征的默认范围（最小值和最大值）
        self.feature_ranges = {
            col: (x[col].min(), x[col].max())
            for col in self.features
        }

    def random_generate_samples(self, n_samples: int, custom_ranges: dict = None) -> pd.DataFrame:
        """
        生成新的样本
        
        参数:
        n_samples: int - 要生成的样本数量
        custom_ranges: dict - 自定义范围字典 {特征名: (最小值, 最大值)}
        
        返回:
        pd.DataFrame - 生成的新样本
        """
        if custom_ranges is None:
            custom_ranges = {}

        generated_data = {}

        for feature in self.features:
            # 使用自定义范围或默认范围
            min_val, max_val = custom_ranges.get(
                feature,
                self.feature_ranges[feature]
            )
            # 生成随机数
            generated_data[feature] = np.random.uniform(
                min_val,
                max_val,
                n_samples
            )

        return pd.DataFrame(generated_data, columns=self.features)

    def grid_generate_samples(self, n_samples: int, custom_ranges: dict = None,
                              density_factor: float = 1.0) -> pd.DataFrame:
        """
        多维网格均匀采样（总样本数≈n_samples）

        参数:
        n_samples: int - 目标总样本数量（允许生成略多）
        custom_ranges: dict - 自定义范围 {特征名: (最小值, 最大值)}
        density_factor: float - 密度调节因子（1.0=严格最小覆盖，>1增加密度）

        返回:
        pd.DataFrame - 多维网格均匀样本（列顺序与原始数据一致）

        核心算法:
        1. 计算每个特征的采样点数 k = ceil(n_samples^(1/d) * density_factor)
        2. 生成k^d个网格点（覆盖所有特征组合）
        3. 自动处理单值特征（不参与网格生成）
        4. 保证：k^d ≥ n_samples（允许最多10%超量）
        """
        if custom_ranges is None:
            custom_ranges = {}
        d = len(self.features)  # 特征维度

        # 处理单特征场景
        if d == 0:
            raise ValueError("DataFrameGenerator: 单特征场景不支持网格生成")
        if d == 1:
            return self._single_feature_grid(n_samples, custom_ranges)

        # 计算每个特征的采样点数
        valid_features = [f for f in self.features
                          if not np.isclose(*custom_ranges.get(f, self.feature_ranges[f]))]
        effective_d = len(valid_features)  # 有效变化的特征数

        if effective_d == 0:
            return pd.DataFrame({f: [custom_ranges.get(f, self.feature_ranges[f])[0]] * n_samples
                                 for f in self.features})

        # 计算最小k值（k^effective_d ≥ n_samples）
        k = max(2, ceil((n_samples ** (1 / effective_d)) * density_factor))
        while (k - 1) ** effective_d >= n_samples:  # 防止过度计算
            k -= 1

        # 生成各特征的采样点
        grid_points = []
        for feature in self.features:
            min_val, max_val = custom_ranges.get(feature, self.feature_ranges[feature])
            if np.isclose(min_val, max_val):
                grid_points.append(np.full(k, min_val))
                continue
            # 生成等间隔点（包含端点，避免边界值缺失）
            points = np.linspace(min_val, max_val, num=k, endpoint=True)
            grid_points.append(points)

        # 生成多维网格（仅处理有效变化的特征）
        mesh = np.meshgrid(*[gp for gp in grid_points if len(np.unique(gp)) > 1])
        if not mesh:  # 所有特征固定值
            return pd.DataFrame([{f: grid_points[i][0] for i, f in enumerate(self.features)}])

        # 展平网格并组合特征
        samples = np.column_stack([m.ravel() for m in mesh])
        fixed_features = [f for i, f in enumerate(self.features)
                          if len(np.unique(grid_points[i])) == 1]

        # 构建完整样本
        full_samples = []
        for point in samples:
            row = {}
            ptr = 0
            for i, f in enumerate(self.features):
                if len(np.unique(grid_points[i])) == 1:
                    row[f] = grid_points[i][0]
                else:
                    row[f] = point[ptr]
                    ptr += 1
            full_samples.append(row)

        # 转换为DataFrame（可能包含k^effective_d个样本）
        df = pd.DataFrame(full_samples)
        return df.reindex(columns=self.features)

    def _single_feature_grid(self, n_samples: int, custom_ranges: dict) -> pd.DataFrame:
        """单特征特殊处理（避免meshgrid退化）"""
        feature = self.features[0]
        min_val, max_val = custom_ranges.get(feature, self.feature_ranges[feature])
        if np.isclose(min_val, max_val):
            return pd.DataFrame({feature: [min_val] * n_samples})
        # 保证至少2个点（避免linspace错误）
        k = max(2, n_samples)
        points = np.linspace(min_val, max_val, num=k, endpoint=True)
        if k > n_samples:  # 截断到目标数量（单特征允许精确控制）
            points = points[:n_samples]
        return pd.DataFrame({feature: points})


class TestDataFrameGenerator(unittest.TestCase):
    def test_generate_samples(self):
        # 创建一个DataFrameGenerator对象
        X = pd.DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6]})
        gen = DataFrameGenerator(X)
        grid_samples = gen.grid_generate_samples(200)
        print(grid_samples.shape)  # (100, 2) （精确10x10网格）
        print(grid_samples)

    def test_grid_generate_samples_with_fixed_features(self):
        X_fixed = pd.DataFrame({'A': [5, 5], 'B': [1, 3]})
        gen_fixed = DataFrameGenerator(X_fixed)
        fixed_samples = gen_fixed.grid_generate_samples(20)
        print(fixed_samples['A'].unique())  # [5] （A固定）
        print(fixed_samples.shape)
        print(fixed_samples)

    def test_random_sample(self):
        X = pd.DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6]})
        gen = DataFrameGenerator(X)
        random_samples = gen.random_generate_samples(100)
        print(random_samples)
