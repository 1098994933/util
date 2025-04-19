import pytest
import pandas as pd
import numpy as np
from ml.FeatureSelection import FeatureSelector


class TestFeatureSelector(object):
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        # 创建具有相关性的特征
        np.random.seed(42)
        n_samples = 100

        # 基础特征
        x1 = np.random.normal(0, 1, n_samples)
        x2 = x1 + np.random.normal(0, 0.1, n_samples)  # x2与x1高度相关
        x3 = x1 + np.random.normal(0, 0.2, n_samples)  # x3与x1高度相关
        x4 = np.random.normal(0, 1, n_samples)  # x4与x1无关
        x5 = x4 + np.random.normal(0, 0.1, n_samples)  # x5与x4高度相关
        x6 = np.random.normal(0, 1, n_samples)
        # 目标变量
        y = 2 * x1 + 3 * x4 + np.random.normal(0, 0.1, n_samples)

        # 创建DataFrame
        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5,
            "x6": x6
        })

        return df, y

    def test_select_features_by_pcc_basic(self, sample_data):
        """测试基本的特征选择功能"""
        X, y = sample_data
        selector = FeatureSelector(X, y, select_method='pcc')

        # 使用较高的相关系数阈值
        selector_params = {'pcc': 0.95}
        selected_features = selector.select_features_by_pcc(X, y, selector_params)
        print(selector.selector_result)
        # 验证结果
        assert len(selected_features) > 0
        assert len(selected_features) < X.shape[1]
        assert 'x1' in selected_features or 'x4' in selected_features

        # 验证selector_result
        result = selector.selector_result
        assert 'selected_features' in result
        assert 'dropped_features' in result
        assert 'pcc_result' in result
        assert 'pcc_threshold' in result
        assert result['pcc_threshold'] == 0.95

    def test_select_features_by_pcc_correlation_records(self, sample_data):
        """测试特征相关性记录功能"""
        X, y = sample_data
        selector = FeatureSelector(X, y, select_method='pcc')
        
        # 使用较高的相关系数阈值
        selector_params = {'pcc': 0.95}
        selector.select_features_by_pcc(X, y, selector_params)
        
        # 验证pcc_result
        pcc_result = selector.selector_result['pcc_result']
        
        # 检查每个保留特征是否都有对应的被删除特征
        for kept_feature in selector.selector_result['selected_features']:
            assert kept_feature in pcc_result
            assert len(pcc_result[kept_feature]) > 0
            
            # 验证被删除特征确实与保留特征高度相关
            for dropped_info in pcc_result[kept_feature]:
                assert 'feature' in dropped_info
                assert 'correlation' in dropped_info
                assert dropped_info['correlation'] >= 0.95
                assert isinstance(dropped_info['correlation'], float)
                
                # 验证相关系数计算正确
                expected_correlation = abs(X[kept_feature].corr(X[dropped_info['feature']]))
                assert abs(dropped_info['correlation'] - expected_correlation) < 1e-10

    def test_select_features_by_pcc_threshold(self, sample_data):
        """测试不同相关系数阈值的影响"""
        X, y = sample_data

        # 使用不同的阈值
        thresholds = [0.8, 0.9, 0.95, 0.99]
        num_selected_features = []

        for threshold in thresholds:
            selector = FeatureSelector(X, y, select_method='pcc')
            selector_params = {'pcc': threshold}
            selected_features = selector.select_features_by_pcc(X, y, selector_params)
            num_selected_features.append(len(selected_features))

        # 验证阈值越高，选择的特征越多
        assert all(num_selected_features[i] <= num_selected_features[i + 1]
                   for i in range(len(num_selected_features) - 1))

    def test_select_features_by_pcc_zero_variance(self, sample_data):
        """测试处理零方差特征的情况"""
        X, y = sample_data

        # 添加零方差特征
        X['zero_var'] = 0

        selector = FeatureSelector(X, y, select_method='pcc')
        selector_params = {'pcc': 0.95}
        selected_features = selector.select_features_by_pcc(X, y, selector_params)

        # 验证零方差特征被移除
        assert 'zero_var' not in selected_features

    def test_select_features_by_pcc_y_input_types(self, sample_data):
        """测试不同类型的y输入"""
        X, y = sample_data
        
        # 测试pd.Series输入
        selector1 = FeatureSelector(X, pd.Series(y), select_method='pcc')
        selector_params = {'pcc': 0.95}
        result1 = selector1.select_features_by_pcc(X, pd.Series(y), selector_params)
        
        # 测试np.ndarray输入
        selector2 = FeatureSelector(X, np.array(y), select_method='pcc')
        result2 = selector2.select_features_by_pcc(X, np.array(y), selector_params)
        
        # 测试pd.DataFrame输入
        selector3 = FeatureSelector(X, pd.DataFrame(y), select_method='pcc')
        result3 = selector3.select_features_by_pcc(X, pd.DataFrame(y), selector_params)
        
        # 验证所有结果相同
        assert result1 == result2 == result3
        
        # 测试错误的多列输入
        with pytest.raises(ValueError):
            selector4 = FeatureSelector(X, pd.DataFrame({'y1': y, 'y2': y}), select_method='pcc')
            selector4.select_features_by_pcc(X, pd.DataFrame({'y1': y, 'y2': y}), selector_params)
            
        with pytest.raises(ValueError):
            selector5 = FeatureSelector(X, np.array([y, y]).T, select_method='pcc')
            selector5.select_features_by_pcc(X, np.array([y, y]).T, selector_params)

    def test_pcc_result_to_dataframe(self, sample_data):
        """测试pcc_result转换为DataFrame的功能"""
        X, y = sample_data
        selector = FeatureSelector(X, y, select_method='pcc')
        
        # 运行特征选择
        selector_params = {'pcc': 0.95}
        selector.select_features_by_pcc(X, y, selector_params)
        
        # 转换为DataFrame
        df = selector.pcc_result_to_dataframe()
        print(df)
        # 验证DataFrame的结构
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {'kept_feature', 'correlated_features', 'correlations'}
        

