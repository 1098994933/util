from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def preprocessing_dataset(df):
    """
    :return:
    """
    features = list(df.columns)
    scale = MinMaxScaler()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(df)
    X = pd.DataFrame(scale.fit_transform(X), columns=features)
    return X, scale, imp


def merge_by_time_and_columns(df1: pd.DataFrame,
                              df2: pd.DataFrame,
                              time_col: str,
                              start_time_col: str,
                              end_time_col: str,
                              merge_columns: List[str],
                              how: str = 'left',
                              time_format: str = None) -> pd.DataFrame:
    """
    基于时间和时间区间和merge DataFrame
    
    参数:
    df1: pd.DataFrame, 第一个DataFrame，包含时间列
    df2: pd.DataFrame, 第二个DataFrame，包含时间开始和结束时间
    time_col: str, df1中的时间列名
    start_time_col: str, df2中的开始时间列名
    end_time_col: str, df2中的结束时间列名
    merge_columns: List[str], 需要匹配的列名列表
    how: str, 关联方式，可选 'inner', 'left', 'right', 'outer'
    time_format: str, 时间格式，如果为None则自动推断
    
    返回:
    pd.DataFrame: 关联后的DataFrame
    """
    # 确保时间列是datetime类型
    if time_format:
        df1[time_col] = pd.to_datetime(df1[time_col], format=time_format)
        df2[start_time_col] = pd.to_datetime(df2[start_time_col], format=time_format)
        df2[end_time_col] = pd.to_datetime(df2[end_time_col], format=time_format)
    else:
        df1[time_col] = pd.to_datetime(df1[time_col])
        df2[start_time_col] = pd.to_datetime(df2[start_time_col])
        df2[end_time_col] = pd.to_datetime(df2[end_time_col])

    # 创建临时列用于关联
    df1['_temp_key'] = 1
    df2['_temp_key'] = 1

    # 执行笛卡尔积
    merged = pd.merge(df1, df2, on=['_temp_key'] + merge_columns, how=how)

    # 根据关联方式处理时间区间匹配
    if how == 'left':
        # 对于left join，保留所有df1的行
        # 创建时间匹配标记
        merged['_time_match'] = (merged[time_col] >= merged[start_time_col]) & (
                merged[time_col] <= merged[end_time_col])

        # 对于没有时间匹配的行，将df2的列设为NaN
        for col in df2.columns:
            if col not in [start_time_col, end_time_col] + merge_columns:
                merged.loc[~merged['_time_match'], col] = np.nan

        # 删除临时列
        merged.drop(['_temp_key', '_time_match'], axis=1, inplace=True)

    else:
        # 对于其他关联方式，只保留时间区间内的记录
        mask = (merged[time_col] >= merged[start_time_col]) & (merged[time_col] <= merged[end_time_col])
        merged = merged[mask].copy()
        merged.drop('_temp_key', axis=1, inplace=True)

    # 处理重复列名
    if how in ['left', 'inner']:
        # 保留df1的列名
        for col in merge_columns:
            if f'{col}_x' in merged.columns:
                merged[col] = merged[f'{col}_x']
                merged.drop([f'{col}_x', f'{col}_y'], axis=1, inplace=True)
    elif how == 'right':
        # 保留df2的列名
        for col in merge_columns:
            if f'{col}_y' in merged.columns:
                merged[col] = merged[f'{col}_y']
                merged.drop([f'{col}_x', f'{col}_y'], axis=1, inplace=True)

    return merged


def merge_with_nearest_time_fill(df1: pd.DataFrame,
                               df2: pd.DataFrame,
                               merge_columns: List[str],
                               time_col_df1: str,
                               time_col_df2: str,
                               fill_columns: List[str] = None,
                               time_format: str = None) -> pd.DataFrame:
    """
    基于指定列关联两个DataFrame，并使用时间最邻近的样本填充缺失值
    
    参数:
    df1: pd.DataFrame, 第一个DataFrame，包含时间列
    df2: pd.DataFrame, 第二个DataFrame，包含时间列
    merge_columns: List[str], 需要匹配的列名列表
    time_col_df1: str, df1中的时间列名
    time_col_df2: str, df2中的时间列名
    fill_columns: List[str], 需要填充的列名列表，如果为None则填充df2中除时间列和合并列外的所有列
    time_format: str, 时间格式，如果为None则自动推断
    
    返回:
    pd.DataFrame: 关联后的DataFrame
    """
    # 确保时间列是datetime类型
    if time_format:
        df1[time_col_df1] = pd.to_datetime(df1[time_col_df1], format=time_format)
        df2[time_col_df2] = pd.to_datetime(df2[time_col_df2], format=time_format)
    else:
        df1[time_col_df1] = pd.to_datetime(df1[time_col_df1])
        df2[time_col_df2] = pd.to_datetime(df2[time_col_df2])

    # 如果fill_columns为None，则填充df2中除时间列和合并列外的所有列
    if fill_columns is None:
        fill_columns = [col for col in df2.columns 
                       if col not in [time_col_df2] + merge_columns]

    # 首先执行普通的left join
    merged = pd.merge(df1, df2, on=merge_columns, how='left')

    # 找出需要填充的行
    missing_mask = merged[fill_columns].isna().any(axis=1)
    if not missing_mask.any():
        return merged

    # 对每个需要填充的列进行处理
    for col in fill_columns:
        # 获取需要填充的行
        missing_rows = merged[missing_mask].copy()

        # 对每个需要填充的行，找到时间最邻近的样本
        for idx in missing_rows.index:
            current_time = merged.loc[idx, time_col_df1]  # 使用df1的时间列

            # 获取相同merge_columns值的样本
            same_group = df2[df2[merge_columns].apply(
                lambda x: all(x == merged.loc[idx, merge_columns]), axis=1
            )].copy()

            if len(same_group) == 0:
                # 如果没有相同merge_columns的样本，则使用所有样本中时间最邻近的
                same_group = df2.copy()

            # 计算时间差
            same_group['_time_diff'] = abs(same_group[time_col_df2] - current_time)

            # 找到时间最邻近的样本
            nearest_idx = same_group['_time_diff'].idxmin()
            nearest_value = same_group.loc[nearest_idx, col]

            # 填充缺失值
            merged.loc[idx, col] = nearest_value

    # 重命名时间列
    merged = merged.rename(columns={time_col_df1: 'time'})
    if time_col_df2 in merged.columns:
        merged = merged.drop(time_col_df2, axis=1)

    return merged
