"""
conditional VAE
"""
import torch
from util.deep_learning.VAE.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class ConditionalVAE(BaseVAE):
    """
    model architecture input x = (B, input_dim) y = onehot([1,0,0])
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim

        self.embed_class = nn.Linear(num_classes, input_dim)
        self.embed_data = nn.Linear(input_dim, input_dim)

        modules = []
        #
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        input_dim += 1  # To account for the extra label
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
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    # nn.ConvTranspose2d(hidden_dims[i],
                    #                    hidden_dims[i + 1],
                    #                    kernel_size=3,
                    #                    stride = 2,
                    #                    padding=1,
                    #                    output_padding=1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_dims[-1], input_dim),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.input_dim).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim=1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               labels: Tensor,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param labels:
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        labels = labels.float()
        z = torch.randn(num_samples, self.latent_dim)

        z = torch.cat([z, labels], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input x, returns the reconstructed x_hat
        :param x: (Tensor) [B x ]
        :return: (Tensor) [B x ]
        """
        return self.forward(x, **kwargs)[0]

