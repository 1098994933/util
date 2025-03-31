"""
classification task
"""

import pickle
import unittest

import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .hyperopt import HyperParameterSearch

from eval import cv, cal_cls_metric


class ClassificationTask(BaseEstimator):
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 models: list,
                 cv: int = None,
                 test_size: float = None,
                 random_state: int = 42,
                 hyperopt: bool = True):
        """
        可配置分类任务框架（支持动态算法列表）

        参数:
        X: 特征矩阵 (pd.DataFrame)
        y: 标签 (pd.Series)
        models: 算法列表（元素为模型配置字典）
                格式: [
                    {'estimator': estimator, 'param_grid': param_grid（可选）},
                    或直接传入 estimator （使用默认参数）
                ]
        cv: 交叉验证折数（None=禁用交叉验证）
        test_size: 测试集比例（None=不划分测试集）
        random_state: 随机种子
        hyperopt: 是否开启超参数优化（默认True）
        """
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.models = self._parse_models(models)  # 解析模型配置
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.hyperopt = hyperopt
        self._split_data()
        self.results = self._init_results()
        
        # 初始化超参数搜索器
        if self.hyperopt:
            self.hyperopt_searcher = HyperParameterSearch(
                method='bayesian',
                cv=self.cv,
                scoring='f1',
                n_trials=100,
                random_state=self.random_state
            )

    def _parse_models(self, models):
        """解析模型配置（支持简写和默认参数）"""
        default_params = {
            LogisticRegression: {'max_iter': 1000},
            RandomForestClassifier: {'n_jobs': -1}
        }
        parsed = []
        for model in models:
            if isinstance(model, BaseEstimator):
                # 直接传入模型（使用默认参数）
                parsed.append({
                    'estimator': model,
                    'param_grid': None,
                    'fixed_params': {}
                })
            else:
                # 解析配置字典
                estimator = model['estimator']
                param_grid = model.get('param_grid')
                fixed_params = model.get('fixed_params', {})
                # 应用默认参数
                if type(estimator) in default_params:
                    estimator.set_params(**default_params[type(estimator)])
                parsed.append({
                    'estimator': estimator.set_params(**fixed_params),
                    'param_grid': param_grid,
                    'fixed_params': fixed_params
                })
        return parsed

    def _split_data(self):
        """split dataset by test size"""
        if self.test_size is None:
            self.X_train = self.X
            self.y_train = self.y
            self.X_test = None
            self.y_test = None
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                test_size=self.test_size,
                stratify=self.y,
                random_state=self.random_state
            )

    def _init_results(self):
        """初始化结果字典（兼容不同模式）"""
        return {
            'dataset': {
                'n_samples': len(self.X),
                'n_features': len(self.X.columns),
                'class_dist': self.y.value_counts(normalize=True).to_dict()
            },
            'model_records': [],
            'final_model': None
        }

    def fit_cls_model(self):
        """execute the task"""
        for model_cfg in self.models:
            estimator = model_cfg['estimator']
            param_grid = model_cfg['param_grid']
            model_name = type(estimator).__name__
            # hyperopt in training set
            if self.hyperopt and param_grid is not None:
                # 使用 HyperParameterSearch 进行贝叶斯优化
                search_results = self.hyperopt_searcher.search(
                    X=self.X_train,
                    y=self.y_train,
                    model_name=model_name,
                    param_grid=param_grid
                )
                best_estimator = estimator.set_params(**search_results['best_params'])
                best_estimator.fit(self.X_train, self.y_train)
                estimator = best_estimator
            else:
                # train 
                estimator.fit(self.X_train, self.y_train)
            y_train_predict = estimator.predict(self.X_train)
            metrics_dict = cal_cls_metric(self.y_train, y_train_predict)
            # training results
            self.results[model_name]['train'] = {
                "y_predict": y_train_predict,
                "y_true": self.y_train,
                "metrics": metrics_dict,
            }
            if self.test_size:
                y_test_predict = estimator.predict(self.X_test)
                metrics_dict = cal_cls_metric(self.y_test, y_test_predict)
                self.results[model_name]['test'] = {
                    "y_predict": y_test_predict,
                    "y_true": self.y_test,
                    "metrics": metrics_dict,
                }
            if self.cv is not None:
                y_cv_predict = cv(estimator, self.X_train, self.y_train, k=self.cv)
                metrics_dict = cal_cls_metric(self.y_train, y_cv_predict)
                self.results[model_name]['cv'] = {
                    "y_predict": y_cv_predict,
                    "y_true": self.y_train,
                    "metrics": metrics_dict,
                }

            # 选择最优模型（by score）
            self._select_best_model(self.results[model_name])

        return self

    def _build_model_record(self, model_name, estimator, cv_scores):
        """构建统一的模型记录（兼容不同验证模式）"""
        record = {
            'model': model_name,
            'params': dict(estimator.get_params()),
            'train_metrics': self._evaluate(estimator, self.X_train, self.y_train)
        }
        if self.cv:
            record['cv_metrics'] = cv_scores
        if self.X_test is not None:
            record['test_metrics'] = self._evaluate(estimator, self.X_test, self.y_test)
        return record

    def _evaluate(self, estimator, X, y):
        """统一评估方法（支持概率输出）"""
        y_pred = estimator.predict(X)
        return {
            **classification_report(y, y_pred, output_dict=True, zero_division=0),
            'probability': estimator.predict_proba(X).mean(axis=0).tolist()
        }

    def _select_best_model(self, scoring='f1_weighted'):
        """根据 score 选择最佳模型"""
        for model_name, model_results in self.results.items():
            current_score = model_results[model_name]['train']['metrics'][scoring]
            if self.results['final_model'] is None or \
                    current_score > self.results['final_model']["cv"]["metrics"][scoring]:
                # 更新最佳模型
                self.results['final_model'] = self.results[model_name]

    
    def save_results(self, path: str):
        """保存结果（不含模型对象）"""
        safe_results = {k: v for k, v in self.results.items() if k != 'final_model'}
        with open(f'{path}_results.pkl', 'wb') as f:
            pickle.dump(safe_results, f)

    def save_model(self, path: str):
        """保存最佳模型"""
        if self.results['final_model']:
            joblib.dump(
                self.results['final_model']['estimator'],
                f'{path}_model.pkl'
            )

    @property
    def summary(self) -> pd.DataFrame:
        """生成结果摘要（兼容不同模式）"""
        df = pd.DataFrame(self.results['model_records']).drop(
            columns=['estimator', 'params'], errors='ignore'
        )
        metrics_cols = ['train_metrics.weighted avg.f1-score']
        if self.cv:
            metrics_cols.append('cv_metrics.mean_f1')
        return df[['model'] + metrics_cols].round(3)



class TestClassificationTask(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        # 加载iris数据集
        iris = load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.y = pd.Series(iris.target)
        
        # 数据标准化
        scaler = StandardScaler()
        self.X = pd.DataFrame(
            scaler.fit_transform(self.X),
            columns=self.X.columns
        )
        
        # 定义测试用的模型列表
        self.models = [
            LogisticRegression(random_state=42),
            RandomForestClassifier(random_state=42),
            XGBClassifier(random_state=42)
        ]
        
        # 创建分类任务实例
        self.task = ClassificationTask(
            X=self.X,
            y=self.y,
            models=self.models,
            cv=3,
            test_size=0.2,
            random_state=42
        )


    def test_model_training(self):
        """测试模型训练"""
        self.task.fit_cls_model()
        print(self.task.results)

        # 检查结果记录
        self.assertIsNotNone(self.task.results)
        self.assertIsNotNone(self.task.results['model_records'])
        self.assertIsNotNone(self.task.results['final_model'])
        # 检查每个模型的训练结果
        for record in self.task.results['model_records']:
            self.assertIn('model', record)
            self.assertIn('train_metrics', record)
            self.assertIn('test_metrics', record)
            
            # 检查评估指标
            metrics = record['train_metrics']
            self.assertIn('weighted avg', metrics)
            self.assertIn('f1-score', metrics['weighted avg'])

    def test_model_selection(self):
        """测试最佳模型选择"""
        self.task.fit_cls_model()
        
        # 检查最佳模型记录
        best_model = self.task.results['final_model']
        print(best_model)

        self.assertIsNotNone(best_model)
        self.assertIn('model', best_model)
        self.assertIn('params', best_model)
        
        # 验证最佳模型是训练过的模型之一
        model_names = [record['model'] for record in self.task.results['model_records']]
        self.assertIn(best_model['model'], model_names)

    def test_results_saving(self):
        """测试结果保存功能"""
        import os
        import tempfile
        
        self.task.fit_cls_model()
        
        # 创建临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试保存结果
            results_path = os.path.join(temp_dir, 'test_results')
            self.task.save_results(results_path)
            self.assertTrue(os.path.exists(f'{results_path}_results.pkl'))
            
            # 测试保存模型
            model_path = os.path.join(temp_dir, 'test_model')
            self.task.save_model(model_path)
            self.assertTrue(os.path.exists(f'{model_path}_model.pkl'))

    def test_summary_property(self):
        """测试结果摘要属性"""
        self.task.fit_cls_model()
        
        # 获取摘要
        summary = self.task.summary
        
        # 检查摘要格式
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('model', summary.columns)
        self.assertIn('train_metrics.weighted avg.f1-score', summary.columns)
        self.assertIn('test_metrics.weighted avg.f1-score', summary.columns)
        
        # 检查摘要内容
        self.assertEqual(len(summary), len(self.models))
        self.assertTrue(all(summary['train_metrics.weighted avg.f1-score'] >= 0))
        self.assertTrue(all(summary['test_metrics.weighted avg.f1-score'] >= 0))

if __name__ == '__main__':
    unittest.main()
        