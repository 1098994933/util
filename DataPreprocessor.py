"""
预处理数据类
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import pickle


class DataPreprocessor(object):
    def __init__(self):
        self.label_encoders = {}
        self.imputer = None
        self.columns_to_drop = []

    def fit(self, df):
        """
        训练数据预处理器：编码字符串列，确定要删除的列，训练缺失值填充器。
        """
        # 删除缺失值大于90%的列
        missing_percentage = df.isnull().mean()
        self.columns_to_drop = missing_percentage[missing_percentage > 0.9].index.tolist()
        df.drop(columns=self.columns_to_drop, inplace=True)

        # 对输入的dataframe中的str列进行LabelEncode编码
        str_columns = df.select_dtypes(include='object').columns
        for column in str_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            self.label_encoders[column] = le

        # KNN填充缺失值
        self.imputer = KNNImputer(n_neighbors=1)
        df_filled = self.imputer.fit_transform(df)
        df[:] = df_filled

    def transform(self, df):
        """
        使用训练好的编码器和填充器转换新数据
        """
        # 删除已知要删除的列
        df.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

        # 编码字符串列
        for column, le in self.label_encoders.items():
            if column in df.columns:
                df[column] = le.transform(df[column].astype(str))

        # KNN填充缺失值
        df_filled = self.imputer.transform(df)
        return pd.DataFrame(df_filled, columns=df.columns)

    def save(self, filepath):
        """
        将预处理器的状态保存到文件
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        """
        从文件加载预处理器
        """
        with open(filepath, 'rb') as file:
            return pickle.load(file)


# 使用示例
# 初始化并训练预处理器
preprocessor = DataPreprocessor()
preprocessor.fit(df_train)

# 保存预处理器
preprocessor.save('preprocessor.pkl')

# 加载预处理器并转换新数据
loaded_preprocessor = DataPreprocessor.load('preprocessor.pkl')
new_df = loaded_preprocessor.transform(df_new)
