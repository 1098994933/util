import unittest
import pandas as pd
from sklearn.impute import KNNImputer

from DataPreprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'col2': ['a', 'b'] * 5,
            'col3': [None]*10,
            'col4': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        self.preprocessor = DataPreprocessor()

    def test_fit(self):
        self.preprocessor.fit(self.df)
        self.assertEqual(isinstance(self.preprocessor.imputer, KNNImputer), True)
        self.assertEqual(len(self.preprocessor.label_encoders), 1)

    def test_transform(self):
        self.preprocessor.fit(self.df)
        transformed_df = self.preprocessor.transform(self.df)
        print(transformed_df)
        self.assertEqual(isinstance(transformed_df, pd.DataFrame), True)

    def test_save_and_load(self):
        self.preprocessor.fit(self.df)
        self.preprocessor.save('test_preprocessor.pkl')
        loaded_preprocessor = DataPreprocessor.load('test_preprocessor.pkl')
        self.assertEqual(isinstance(loaded_preprocessor, DataPreprocessor), True)


if __name__ == '__main__':
    unittest.main()
