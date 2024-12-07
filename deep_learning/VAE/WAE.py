import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from util.deep_learning.VAE.base import OneDimensionalDataset


class WAE(nn.Module):
    """
    Wasserstein Autoencoder with Maximum Mean Discrepancy (MMD) for 1D data.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List = None, reg_weight: int = 100,
                 kernel_type: str = 'imq', latent_var: float = 2., **kwargs):
        """
        :param input_dim: Input dimension
        :param latent_dim: Latent space dimension
        :param hidden_dims: List of hidden dimensions
        :param reg_weight: Regularization weight
        :param kernel_type: Kernel type for MMD
        :param latent_var: Variance of the latent space
        """
        super(WAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            input_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1], latent_dim)
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], self.input_dim),
                            nn.Sigmoid())  # Assuming the data is normalized between 0 and 1

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        z = self.fc_z(result)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        recons_loss = F.mse_loss(recons, input)

        mmd_loss = self.compute_mmd(z, self.reg_weight)

        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'MMD': mmd_loss}

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        result = kernel.sum() - kernel.diag().sum()
        return result

    def compute_mmd(self, z: torch.Tensor, reg_weight: float) -> torch.Tensor:
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = reg_weight * (prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean())
        return mmd

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd


class WAETrainer(object):
    """
    Trainer class for Wasserstein Autoencoder (WAE) with 1D data.
    """
    def __init__(self, data, input_dim: int, latent_dim: int, batch_size: int = 128, learning_rate=0.0001,val_size=0.2):
        self.data = data
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Split data into training and validation sets
        n = len(data)
        train_size = int((1-val_size) * n)
        val_size = n - train_size
        self.train_data, self.val_data = random_split(data, [train_size, val_size])

        self.model = WAE(input_dim, latent_dim)  # Initialize the WAE model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.best_val_loss = float('inf')
        self.patience = 50
        self.epochs_without_improvement = 0

    def loss_function(self, recon_x, x, z):
        # Reconstruction loss
        recons_loss = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction='sum')
        # MMD loss
        mmd_loss = self.model.compute_mmd(z, self.model.reg_weight)
        return recons_loss + mmd_loss

    def train(self, epochs):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                recon_batch, batch, z = self.model(batch)
                loss = self.loss_function(recon_batch, batch, z)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_data)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    recon_batch, batch, z = self.model(batch)
                    loss = self.loss_function(recon_batch, batch, z)
                    val_loss += loss.item()

            val_loss /= len(self.val_data)

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save the model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.epochs_without_improvement += 1
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    def generate_samples(self, num_samples: int):
        """
        Generate samples from the model.
        """
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.model.decode(z)
        return samples

    def reconstruct(self, x: torch.Tensor):
        """
        Reconstruct the input data.
        """
        self.model.eval()
        with torch.no_grad():
            _, z = self.model.encode(x)
            recon_x = self.model.decode(z)
        return recon_x

    def merge_val_data_and_reconstruct(self):
        self.model.eval()
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)

        val_data_list = []
        recon_data_list = []

        with torch.no_grad():
            for batch in val_loader:
                recon_batch = self.reconstruct(batch)
                val_data_list.append(batch.cpu().numpy())
                recon_data_list.append(recon_batch.cpu().numpy())

        val_data_combined = np.concatenate(val_data_list, axis=0)
        recon_data_combined = np.concatenate(recon_data_list, axis=0)

        val_df = pd.DataFrame(val_data_combined, columns=[f"Feature_{i+1}" for i in range(self.input_dim)])
        recon_df = pd.DataFrame(recon_data_combined, columns=[f"Reconstructed_Feature_{i+1}" for i in range(self.input_dim)])

        merged_df = pd.concat([val_df, recon_df], axis=1)
        return merged_df


