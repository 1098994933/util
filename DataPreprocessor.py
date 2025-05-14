import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import pickle
import unittest


class DataPreprocessor(object):
    def __init__(self, missing_threshold=0.9, n_neighbors=1):
        # 存储每个字符串列对应的编码器
        self.label_encoders = {}
        # 用于填充缺失值的 KNN 填充器
        self.imputer = None
        # 存储需要删除的列名
        self.columns_to_drop = []
        # 缺失值占比阈值，超过该阈值的列将被删除
        self.missing_threshold = missing_threshold
        # KNN 填充器的邻居数量
        self.n_neighbors = n_neighbors

    def fit(self, df):
        """
        训练数据预处理器：编码字符串列，确定要删除的列，训练缺失值填充器。
        """
        try:
            # 删除缺失值大于指定阈值的列
            missing_percentage = df.isnull().mean()
            self.columns_to_drop = missing_percentage[missing_percentage > self.missing_threshold].index.tolist()
            df = df.drop(columns=self.columns_to_drop)

            # 对输入的 dataframe 中的 str 列进行 LabelEncode 编码
            str_columns = df.select_dtypes(include='object').columns
            for column in str_columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le

            # KNN 填充缺失值
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            df_filled = self.imputer.fit_transform(df)
            df[:] = df_filled
        except Exception as e:
            print(f"在 fit 方法中出现错误: {e}")

    def transform(self, df):
        """
        使用训练好的编码器和填充器转换新数据
        """
        try:
            # 删除已知要删除的列
            df = df.drop(columns=self.columns_to_drop, errors='ignore')

            # 编码字符串列
            for column, le in self.label_encoders.items():
                if column in df.columns:
                    # 处理新标签 if not in the labelEncoder
                    valid_mask = df[column].astype(str).isin(le.classes_)
                    df.loc[~valid_mask, column] = le.classes_[0]
                    df[column] = le.transform(df[column].astype(str))

            # KNN 填充缺失值
            df_filled = self.imputer.transform(df)
            return pd.DataFrame(df_filled, columns=df.columns)
        except Exception as e:
            print(f"在 transform 方法中出现错误: {e}")
            return None

    def save(self, filepath):
        """
        将预处理器的状态保存到文件
        """
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(self, file)
        except Exception as e:
            print(f"保存预处理器时出现错误: {e}")

    @classmethod
    def load(cls, filepath):
        """
        从文件加载预处理器
        """
        try:
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"加载预处理器时出现错误: {e}")
            return None

    @staticmethod
    def delete_outliers_y(df: pd.DataFrame, y_col):
        """
        delete outliers with 3 sigma rule
        return df_clean
        """
        mean = df[y_col].mean()
        std = df[y_col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        is_outlier = (df[y_col] < lower_bound) | (df[y_col] > upper_bound)

        deleted_count = is_outlier.sum()
        df_clean = df[~is_outlier].copy()

        print(f"delete {deleted_count} outlier not in [{lower_bound},{upper_bound}]")
        return df_clean


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'missing_col': [1, 2, None, 4, 5],
            'y_col': [1, 2, 3, 4, 5]
        })
        self.preprocessor = DataPreprocessor()

    def test_delete_outliers_y(self):
        data = {'y': [1, 2, 3, 4, 5, 6, 7, 0, -9, 8, 8, 150]}  # 100 是异常值
        df = pd.DataFrame(data)

        result = self.preprocessor.delete_outliers_y(df, 'y')
        print(result)


