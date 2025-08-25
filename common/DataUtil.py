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


def convert_np(obj):
    """
    convert numpy object to python object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    else:
        return obj


def safe_json_serialize_dict(data_dict: Dict[str, Any], ensure_ascii: bool = False,
                             indent: int = None, float_format: str = '.6g') -> str:
    """
    安全地将字典序列化为JSON字符串，处理numpy类型、无穷大、NaN等特殊值

    Args:
        data_dict: 要序列化的字典
        ensure_ascii: 是否确保ASCII编码，默认False支持中文
        indent: JSON缩进，默认None（不缩进）
        float_format: 浮点数格式化字符串，默认'.6g'

    Returns:
        str: JSON字符串

    Raises:
        TypeError: 当字典无法序列化时抛出

    Example:
        >>> import numpy as np
        >>> data = {
        ...     'array': np.array([1, 2, 3]),
        ...     'float': np.float64(3.14159),
        ...     'inf': np.inf,
        ...     'nan': np.nan,
        ...     'nested': {'value': np.array([4, 5, 6])}
        ... }
        >>> json_str = safe_json_serialize_dict(data, indent=2)
        >>> print(json_str)
        {
          "array": [1, 2, 3],
          "float": 3.14159,
          "inf": null,
          "nan": null,
          "nested": {
            "value": [4, 5, 6]
          }
        }
    """
    import json

    def _convert_value(value):
        """转换单个值"""
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(f"{value:{float_format}}")
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, dict):
            return {k: _convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_convert_value(item) for item in value]
        elif isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(f"{value:{float_format}}")
        elif hasattr(value, 'tolist'):  # 其他可能有tolist方法的对象
            return value.tolist()
        elif hasattr(value, 'item'):  # 其他可能有item方法的对象
            return value.item()
        else:
            return value

    try:
        # 预处理字典
        processed_dict = _convert_value(data_dict)
        # 序列化为JSON
        return json.dumps(processed_dict, ensure_ascii=ensure_ascii, indent=indent)
    except Exception as e:
        raise TypeError(f"无法序列化字典: {str(e)}")


def safe_json_dumps(data_dict: Dict[str, Any], **kwargs) -> str:
    """
    安全的字典JSON序列化快捷方法

    Args:
        data_dict: 要序列化的字典
        **kwargs: 传递给safe_json_serialize_dict的参数

    Returns:
        str: JSON字符串

    Example:
        >>> data = {'array': np.array([1, 2, 3]), 'value': 3.14159}
        >>> json_str = safe_json_dumps(data, indent=2)
    """
    return safe_json_serialize_dict(data_dict, **kwargs)


def build_where_conditions(filters: dict, table_name: str = None) -> tuple:
    """
    根据 filters 字典和表名构建 SQL WHERE 条件和参数（带表名前缀）

    Args:
        filters: 字典，key为字段名称，value为值或list
        table_name: 表名，默认为None，若提供则为字段添加表名前缀

    Returns:
        tuple: (where_clause, params)
            where_clause: SQL WHERE 条件字符串，使用:key或:keyN作为占位符，可带表名前缀
            params: 字典参数，key为占位符名称，value为对应值

    Example:
        filters = {
            "status": "active",
            "category": ["A", "B", "C"],
            "priority": 1
        }
        当table_name="tasks"时返回: (
            "WHERE tasks.status=:status AND tasks.category IN (:category1,:category2,:category3) AND tasks.priority=:priority",
            {
                "status": "active",
                "category1": "A",
                "category2": "B",
                "category3": "C",
                "priority": 1
            }
        )
    """
    if not filters:
        return "", {}

    conditions = []
    params = {}

    # 生成字段前缀（表名 + 点号），如果提供了表名
    field_prefix = f"{table_name}." if table_name else ""

    for field, value in filters.items():
        # 带表名前缀的完整字段名
        full_field = f"{field_prefix}{field}"

        if isinstance(value, list):
            # 处理 list 类型，使用 IN 操作符
            if value:  # 确保 list 不为空
                placeholders = []
                # 为列表中的每个值创建带数字后缀的占位符
                for i, item in enumerate(value, 1):
                    param_key = f"{field}{i}"
                    placeholders.append(f":{param_key}")
                    params[param_key] = item
                conditions.append(f"{full_field} IN ({','.join(placeholders)})")
        else:
            # 处理单个值，使用等号，直接使用field作为参数键
            conditions.append(f"{full_field}=:{field}")
            params[field] = value

    where_clause = " AND ".join(conditions) if conditions else ""

    return where_clause, params