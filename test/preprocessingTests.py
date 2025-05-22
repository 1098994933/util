import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from preprocessing import merge_by_time_and_columns,merge_with_nearest_time_fill


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """创建测试数据"""
        # 时间区间合并的测试数据
        self.df1_time_interval = pd.DataFrame({
            'time': ['2023-01-01 12:00', '2023-01-02 10:00', '2023-01-03 15:00', '2025-01-03 15:00'],
            'product': ['A', 'B', 'C', "C"],
            'value': [100, 200, 300, 300]
        })

        self.df2_time_interval = pd.DataFrame({
            'start_time': ['2023-01-01 08:00', '2023-01-02 09:00', '2023-01-03 14:00'],
            'end_time': ['2023-01-01 18:00', '2023-01-02 17:00', '2026-01-03 16:00'],
            'product': ['A', 'B', 'C'],
            'price': [1000, 2000, 3000]
        })

        # 时间邻近填充的测试数据
        self.df1_nearest_time = pd.DataFrame({
            'id': [1, 2, 3, 7, 5],
            'value': [100, 200, 300, 400, 500],
            'time1': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        })

        self.df2_nearest_time = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'time2': pd.to_datetime(['2023-01-01',
                                   '2023-01-02',
                                   '2023-01-01', '2023-01-03'
                                   ]),
            'price': [1000, 1100, 2000, 2100]
        })

    def test_merge_by_time_and_columns_basic(self):
        """测试时间区间合并的基本功能"""
        result = merge_by_time_and_columns(
            df1=self.df1_time_interval,
            df2=self.df2_time_interval,
            time_col='time',
            start_time_col='start_time',
            end_time_col='end_time',
            merge_columns=['product'],
            how='left',
            time_format='%Y-%m-%d %H:%M'
        )
        print(result)
        
        # 检查结果
        self.assertEqual(len(result), len(self.df1_time_interval))  # 保持df1的行数
        self.assertIn('price', result.columns)  # 包含df2的列
        self.assertFalse(result['product'].isna().any())  # 关联列没有缺失值

        # 检查时间区间匹配
        matched_rows = result[~result['price'].isna()]
        for _, row in matched_rows.iterrows():
            self.assertGreaterEqual(row['time'], row['start_time'])
            self.assertLessEqual(row['time'], row['end_time'])

    def test_merge_with_nearest_time_fill_basic(self):
        """测试时间邻近填充的基本功能"""
        # 打印输入数据
        print("\n输入数据 df1:")
        print(self.df1_nearest_time)
        print("\n输入数据 df2:")
        print(self.df2_nearest_time)

        result = merge_with_nearest_time_fill(
            df1=self.df1_nearest_time,
            df2=self.df2_nearest_time,
            merge_columns=['id'],  # 只使用id进行合并
            time_col_df1='time1',  # df1的时间列名
            time_col_df2='time2',  # df2的时间列名
            fill_columns=None,  # 默认填充所有列
            time_format='%Y-%m-%d'
        )

        # 打印结果
        print("\n合并结果:")
        print(result)

        # 检查结果
        self.assertEqual(len(result), len(self.df1_nearest_time))  # 保持df1的行数
        self.assertIn('price', result.columns)  # 包含df2的列
        self.assertFalse(result['price'].isna().any())  # 所有缺失值都被填充

        # 特别检查id为7的行的填充值
        id_7_row = result[result['id'] == 7].iloc[0]
        print("\nID为7的填充结果:")
        print(id_7_row)

        # 检查填充的值是否在df2的price范围内
        self.assertIn(id_7_row['price'], self.df2_nearest_time['price'].values)


if __name__ == '__main__':
    unittest.main() 