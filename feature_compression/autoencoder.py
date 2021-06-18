import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1028),
            nn.ReLU(),
            nn.Linear(1028, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

loss = nn.MSELoss()

def autoencoder_loss(x_hat, x):
    return loss(x_hat, x)
