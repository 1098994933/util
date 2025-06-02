"""
测试CategorySubGroupDiscovery模块的功能
"""
import unittest
import numpy as np
import pandas as pd
from subgroupDiscovery.CategorySubGroupDiscovery import subgroup_discovery_by_dimensions


class TestCategorySubGroupDiscovery(unittest.TestCase):
    """测试子群发现功能"""

    def setUp(self):
        """准备测试数据"""
        # 生成测试数据
        np.random.seed(42)
        n_samples = 1000
        
        # 生成基础数据
        regions = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
        product_types = np.random.choice(['Electronics', 'Clothing', 'Books'], n_samples)
        user_ages = np.random.randint(18, 65, n_samples)
        browsing_times = np.random.normal(30, 10, n_samples).astype(int)
        
        # 初始化购买金额数组
        purchase_amounts = np.zeros(n_samples)
        
        # 为North地区的Electronics产品添加规律性
        north_electronics_mask = (regions == 'North') & (product_types == 'Electronics')
        # 购买金额与用户年龄和浏览时间正相关
        purchase_amounts[north_electronics_mask] = (
            100 +  # 基础金额
            2 * user_ages[north_electronics_mask] +  # 年龄影响
            5 * browsing_times[north_electronics_mask] +  # 浏览时间影响
            np.random.normal(0, 50, sum(north_electronics_mask))  # 添加一些随机噪声
        )
        
        # 为其他组合生成随机数据
        other_mask = ~north_electronics_mask
        purchase_amounts[other_mask] = np.abs(np.random.normal(500, 200, sum(other_mask)))
        
        # 确保所有购买金额为正
        purchase_amounts = np.abs(purchase_amounts)
        
        self.data = {
            'region': regions,
            'product_type': product_types,
            'user_age': user_ages,
            'browsing_time': browsing_times,
            'purchase_amount': purchase_amounts
        }
        self.df = pd.DataFrame(self.data)
        self.dimension_cols = ['region', 'product_type']
        self.feature_cols = ['user_age', 'browsing_time']
        self.target_col = 'purchase_amount'

    def test_analyze_subgroups_by_dimensions_basic(self):
        """测试基本的子群分析功能"""
        results = subgroup_discovery_by_dimensions(
            df=self.df,
            dimension_cols=self.dimension_cols,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            cv_folder=5
        )
        
        # 验证结果不为空
        self.assertGreater(len(results), 0)
        
        # 验证结果格式
        first_result = results[0]
        self.assertIn('dimensions', first_result)
        self.assertIn('sample_size', first_result)
        self.assertIn('rf_metrics', first_result)
        self.assertIn('dt_metrics', first_result)
        self.assertIn('lr_metrics', first_result)
        self.assertIn('max_r2', first_result)
        
        # 验证评估指标
        for model_metrics in [first_result['rf_metrics'], first_result['dt_metrics'], first_result['lr_metrics']]:
            self.assertIn('MSE', model_metrics)
            self.assertIn('RMSE', model_metrics)
            self.assertIn('MAE', model_metrics)
            self.assertIn('R2', model_metrics)
            self.assertIn('R', model_metrics)

    def test_analyze_subgroups_by_dimensions_min_samples(self):
        """测试最小样本数过滤功能"""
        # 使用较大的最小样本数
        min_samples = 1000  # 设置一个大于数据集大小的值
        results = subgroup_discovery_by_dimensions(
            df=self.df,
            dimension_cols=self.dimension_cols,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            min_samples=min_samples,
            cv_folder=5
        )
        
        # 验证没有结果（因为所有子群都小于最小样本数）
        self.assertEqual(len(results), 0)

    def test_analyze_subgroups_by_dimensions_single_dimension(self):
        """测试单个维度的子群分析"""
        results = subgroup_discovery_by_dimensions(
            df=self.df,
            dimension_cols=['region'],  # 只使用一个维度
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            cv_folder=5
        )
        
        # 验证结果数量应该等于region的唯一值数量
        self.assertEqual(len(results), len(self.df['region'].unique()))
        
        # 验证每个结果的维度字典只包含region
        for result in results:
            self.assertEqual(len(result['dimensions']), 1)
            self.assertIn('region', result['dimensions'])

    def test_analyze_subgroups_by_dimensions_sorting(self):
        """测试结果排序功能"""
        results = subgroup_discovery_by_dimensions(
            df=self.df,
            dimension_cols=self.dimension_cols,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            cv_folder=5
        )
        
        # 验证结果按max_r2降序排序
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]['max_r2'], results[i + 1]['max_r2'])

    def test_analyze_subgroups_by_dimensions_north_electronics(self):
        """测试North地区Electronics产品的规律性"""
        results = subgroup_discovery_by_dimensions(
            df=self.df,
            dimension_cols=self.dimension_cols,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            cv_folder=5
        )
        
        # 找到North地区Electronics产品的结果
        north_electronics_result = None
        for result in results:
            if (result['dimensions'].get('region') == 'North' and 
                result['dimensions'].get('product_type') == 'Electronics'):
                north_electronics_result = result
                break
        
        # 验证North地区Electronics产品的R2值应该较高
        self.assertIsNotNone(north_electronics_result)
        self.assertGreater(north_electronics_result['max_r2'], 0.5)  # 由于有规律性，R2应该较高


if __name__ == '__main__':
    unittest.main() 