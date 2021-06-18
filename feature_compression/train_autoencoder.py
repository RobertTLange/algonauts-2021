import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from autoencoder import autoencoder_loss
# Import utilities for Cluster distributed runs
from mle_toolbox.utils import set_random_seeds

# Minimal VAE implementation
# Adopted from https://github.com/Atcold/pytorch-Deep-Learning/blob/master/11-VAE.ipynb

class Dataset(data.Dataset):
    """ Simple Dataset Wrapper for your Data"""
    def __init__(self, X):
        """ Wrap the data in the dataset torch wrapper """
        self.X = X

    def __len__(self):
        """ Get the number of samples in the buffer """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Get one sample from the dataset"""
        X = self.X[index, ...]
        return X


def fit_autoencoder(net, x_train, batch_size=128, num_epochs=100):
    torch.set_num_threads(10)
    set_random_seeds(1234, verbose=False)

    optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)

    # Set parameters for the dataloaders
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 2}
    training_set = Dataset(x_train)
    data_loader = data.DataLoader(training_set, **train_params)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net = net.to(device)

    net, train_loss = train(net, optimizer, data_loader)

    time_tic = [0]
    stats_tic = [train_loss]
    # Loop over training epochs
    for ep in range(num_epochs):
        net, train_loss = train(net, optimizer, data_loader)
        time_tic.append(ep+1)
        stats_tic.append(train_loss)
    print(stats_tic)
    return net, time_tic, stats_tic


def train(net, optimizer, train_loader):
    net.train()
    train_loss = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for x in train_loader:
        # ===================forward=====================
        x = x.to(device).float()
        z, x_hat= net(x)
        loss = autoencoder_loss(x_hat, x)
        train_loss += loss.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    train_loss /= len(train_loader.dataset)
    return net, train_loss
