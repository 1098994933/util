"""
subgroup discovery by category
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from typing import List, Dict, Tuple
from common.DataUtil import split_dataset_by_dimensions
from eval import cal_reg_metric


def subgroup_discovery_by_dimensions(df: pd.DataFrame, dimension_cols: List[str], feature_cols: List[str], target_col: str, min_samples: int = 30,
                                     cv_folder: int = 5) -> List[Dict]:
    """
    分析数据集中不同维度组合下的子群表现
    
    Args:
        df: 输入的数据集
        dimension_cols: 维度列名列表（字符串类型）
        feature_cols: 特征列名列表（数值类型）
        target_col: 目标列名
        min_samples: 最小样本数要求
        cv_folder: 交叉验证折数
        
    Returns:
        List[Dict]: 包含每个子群分析结果的列表，按最大R2降序排序
    """
    # 1. 对分类变量进行独热编码
    encoder = OneHotEncoder(sparse_output=False)
    cat_features = encoder.fit_transform(df[dimension_cols])
    cat_feature_names = encoder.get_feature_names_out(dimension_cols)

    # 2. 准备特征矩阵
    X = pd.concat([df[feature_cols], pd.DataFrame(cat_features, columns=cat_feature_names)], axis=1)
    y = df[target_col]

    # 3. 使用随机森林和决策树进行特征重要性分析
    models = {'rf': RandomForestRegressor(n_estimators=100, random_state=42), 'dt': DecisionTreeRegressor(random_state=42)}

    # 4. 获取特征重要性
    importance_scores = {}
    for name, model in models.items():
        model.fit(X, y)
        importance_scores[name] = dict(zip(X.columns, model.feature_importances_))

    # 5. 选择重要性最高的维度（最多3个）
    avg_importance = {}
    for feature in cat_feature_names:
        avg_importance[feature] = np.mean([scores[feature] for scores in importance_scores.values()])

    sorted_dimensions = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    print("Sorted Dimensions:", sorted_dimensions)

    # 选择前三个重要维度
    top_dimensions = []
    seen_dimensions = set()
    for dim, importance in sorted_dimensions:
        if len(top_dimensions) >= 3:
            break
        # 获取原始维度名称
        original_dim = None
        for orig_dim in dimension_cols:
            if dim.startswith(orig_dim):
                original_dim = orig_dim
                break
        if original_dim and original_dim not in seen_dimensions:
            top_dimensions.append(original_dim)
            seen_dimensions.add(original_dim)
    
    print(f"Top dimensions: {top_dimensions}")

    selected_dims = list(set(top_dimensions))  # 去重
    print("Selected Dimensions:", selected_dims)

    # 6. 使用选定的维度进行子群分析
    results = []
    for dim_dict, subgroup_df in split_dataset_by_dimensions(df, selected_dims):
        if len(subgroup_df) < min_samples:
            continue

        # 准备子群数据
        X_sub = subgroup_df[feature_cols]
        y_sub = subgroup_df[target_col]

        # 使用交叉验证预测和评估模型性能
        cv_scores = {}

        # 随机森林和决策树模型
        for name, model in models.items():
            y_predict = cross_val_predict(model, X_sub, y_sub, cv=cv_folder)
            metrics = cal_reg_metric(y_sub, y_predict)
            cv_scores[name] = metrics

        # 线性回归模型（使用Pipeline进行标准化）
        lr_pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
        y_predict = cross_val_predict(lr_pipeline, X_sub, y_sub, cv=cv_folder)
        metrics = cal_reg_metric(y_sub, y_predict)
        cv_scores['lr'] = metrics

        # 记录结果
        result = {
            'dimensions': dim_dict,
            'sample_size': len(subgroup_df),
            'rf_metrics': cv_scores['rf'],
            'dt_metrics': cv_scores['dt'],
            'lr_metrics': cv_scores['lr'],
            'max_r2': float(np.max([scores['R2'] for scores in cv_scores.values()]))
        }
        results.append(result)

    # 7. 按最大R2降序排序结果
    results.sort(key=lambda x: x['max_r2'], reverse=True)

    return results
