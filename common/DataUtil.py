"""
数据集工具类
"""

import pandas as pd
import numpy as np
from typing import List, Generator, Tuple, Union, Dict, Any


def split_dataset_by_dimensions(dataset: pd.DataFrame, dimensions: List[str]) -> Generator[Tuple[dict, pd.DataFrame], None, None]:
    """
    根据维度分割数据集，返回一个生成器，每次生成一个维度组合及其对应的数据子集
    
    Args:
        dataset (pd.DataFrame): 输入的数据集
        dimensions (List[str]): 用于分割的维度列名列表
    
    Yields:
        Tuple[dict, pd.DataFrame]: 包含维度组合的字典和对应的数据子集
        - dict: 维度组合的键值对，如 {'dim1': value1, 'dim2': value2}
        - pd.DataFrame: 该维度组合对应的数据子集
    
    Example:
        >>> df = pd.DataFrame({
        ...     'material': ['A', 'A', 'B', 'B'],
        ...     'temperature': [25, 50, 25, 50],
        ...     'value': [1, 2, 3, 4]
        ... })
        >>> for dims, sub_df in split_dataset_by_dimensions(df, ['material', 'temperature']):
        ...     print(dims, sub_df)
        {'material': 'A', 'temperature': 25}   material  temperature  value
        0        A           25      1
        {'material': 'A', 'temperature': 50}   material  temperature  value
        1        A           50      2
        ...
    """
    # 获取所有维度列的唯一值组合
    unique_combinations = dataset[dimensions].drop_duplicates()

    # 遍历每个维度组合
    for _, row in unique_combinations.iterrows():
        # 构建维度组合的字典
        dim_dict = {dim: row[dim] for dim in dimensions}

        # 构建过滤条件
        mask = pd.Series(True, index=dataset.index)
        for dim, value in dim_dict.items():
            mask &= (dataset[dim] == value)

        # 获取对应的数据子集
        sub_dataset = dataset[mask]

        yield dim_dict, sub_dataset
    

def split_dataset_by_frequency(dataset: pd.DataFrame, column: str, bins: int) -> List[pd.DataFrame]:
    """
    对数值型列进行等频率分割
    Args:
        dataset (pd.DataFrame): 输入的数据集
        column (str): 用于分割的数值型列名
        bins (int): 分割的区间数
    Returns:
        List[pd.DataFrame]: 分割后的数据集列表
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    
    if column not in dataset.columns:
        raise KeyError(f"Column '{column}' not found in dataset")
    
    # 获取数值型列
    numeric_column = dataset[column]

    # 计算等频率分割点，返回标签
    labels = pd.qcut(numeric_column, bins, labels=False, duplicates='drop')
    
    # 根据标签分割数据集
    splits = []
    for i in range(bins):
        mask = labels == i
        if mask.any():  # 只添加非空的分割
            splits.append(dataset[mask])

    return splits


def format_float(obj: Union[pd.DataFrame, Dict, List, np.ndarray, float], n: int = 3) -> Union[pd.DataFrame, Dict, List, np.ndarray, float]:
    """
    将输入对象中的浮点数格式化为指定有效数字位数
    
    Args:
        obj: 输入对象，可以是DataFrame、字典、列表、numpy数组或单个浮点数
        n: 保留的有效数字位数，默认为3
    
    Returns:
        格式化后的对象，保持原始类型
    
    Example:
        >>> df = pd.DataFrame({'A': [1.23456, 2.34567], 'B': [3.45678, 4.56789]})
        >>> format_float(df, 3)
           A     B
        0  1.23  3.46
        1  2.35  4.57
        
        >>> format_float({'a': 1.23456, 'b': 2.34567}, 3)
        {'a': 1.23, 'b': 2.35}
        
        >>> format_float([1.23456, 2.34567], 3)
        [1.23, 2.35]
        
        >>> format_float(np.array([1.23456, 2.34567]), 3)
        array([1.23, 2.35])
        
        >>> format_float(1.23456, 3)
        1.23
        
        >>> # 嵌套字典示例
        >>> nested_dict = {
        ...     'a': 1.23456,
        ...     'b': {
        ...         'c': 2.34567,
        ...         'd': [3.45678, 4.56789]
        ...     }
        ... }
        >>> format_float(nested_dict, 3)
        {'a': 1.23, 'b': {'c': 2.35, 'd': [3.46, 4.57]}}
    """
    def _format_single(x: Any) -> Any:
        """处理单个值"""
        if isinstance(x, float):
            return float(f"{x:.{n}g}")
        return x

    if isinstance(obj, pd.DataFrame):
        # 处理DataFrame
        return obj.applymap(_format_single)
    
    elif isinstance(obj, dict):
        # 处理字典（包括嵌套字典）
        return {k: format_float(v, n) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        # 处理列表
        return [format_float(x, n) for x in obj]
    
    elif isinstance(obj, np.ndarray):
        # 处理numpy数组
        return np.vectorize(_format_single)(obj)
    
    elif isinstance(obj, float):
        # 处理单个浮点数
        return _format_single(obj)
    
    else:
        # 其他类型直接返回
        return obj

