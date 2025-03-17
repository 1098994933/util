import unittest
import pandas as pd
from generator.TableDataGenerator import TableDataGenerator


class TestTableDataGenerator(unittest.TestCase):
    def test_generation(self):
        # 创建一个简单的 DataFrame 用于测试
        data = pd.DataFrame({
            'col1': ["a", "b", "c"] * 300,
            'col2': ["a", "b", "c"] * 300,
            'col3': ["d", "e", "f"] * 300,
            'col4': ["d", "e", "f"] * 300,
        })
        generator = TableDataGenerator(data)
        generator.train(epochs=200)
        generated_df = generator.generate(num_samples=100)
        print(generated_df)

    def test_generation2(self):
        # 创建一个简单的 DataFrame 用于测试
        data = pd.DataFrame({
            'col1': [1, 2, 10] * 300,
            'col2': [1, 2, 10] * 300,
            'col3': [1, 2, 10] * 300,
            'col4': [1, 2, 10] * 300,
        })
        generator = TableDataGenerator(data)
        generator.train(epochs=200)
        generated_df = generator.generate(num_samples=100)
        print(generated_df)


if __name__ == "__main__":
    unittest.main()
