"""
Tests case for the deep learning module.
"""
import unittest
import torch
from util.deep_learning.VAE.WAE import WAE


class TestWAE(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的一维数据集
        self.input_dim = 10  # 输入数据的维度
        self.latent_dim = 2  # 潜在空间的维度
        self.reg_weight = 100  # 正则化权重
        self.kernel_type = 'imq'  # 核函数类型
        self.latent_var = 2.  # 潜在空间的方差

        # 生成随机数据
        self.data = torch.randn(100, self.input_dim)  # 100个样本

        # 初始化WAE_MMD模型
        self.model = WAE(self.input_dim, self.latent_dim)

    def test_model_initialization(self):
        # 测试模型是否正确初始化
        self.assertEqual(self.model.latent_dim, self.latent_dim)
        self.assertEqual(self.model.reg_weight, self.reg_weight)
        self.assertEqual(self.model.kernel_type, self.kernel_type)

    def test_forward_pass(self):
        # 测试前向传播
        output, _, _ = self.model(self.data)
        self.assertEqual(output.shape, (100, self.input_dim))

    def test_reconstruction_loss(self):
        # 测试重构损失
        reconstructed, input_data, _ = self.model(self.data)
        loss = self.model.loss_function(reconstructed, input_data, torch.randn_like(reconstructed))
        self.assertGreater(loss['Reconstruction_Loss'], 0)

    def test_mmd_loss(self):
        # 测试MMD损失
        reconstructed, input_data, z = self.model(self.data)
        loss = self.model.loss_function(reconstructed, input_data, z)  # 使用随机z进行测试
        self.assertGreater(loss['MMD'], 0)

    def test_sample_generation(self):
        # 测试样本生成
        num_samples = 10
        samples = self.model.sample(num_samples, torch.device('cpu'))
        self.assertEqual(samples.shape, (num_samples, self.input_dim))



if __name__ == '__main__':
    unittest.main()
