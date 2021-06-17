from autoencoder import VAE, vae_loss
import torch
import torch.nn as nn
import torch.optim as optim
from train import get_data_ready

# Import utilities for Cluster distributed runs
from mle_toolbox.utils import set_random_seeds

# Minimal VAE implementation
# Adopted from https://github.com/Atcold/pytorch-Deep-Learning/blob/master/11-VAE.ipynb


def main(net_config, train_config, log_config):
    torch.set_num_threads(train_config.num_torch_threads)
    # First things first - Set the random seed for the example run
    data_train_loader, data_test_loader, data_train_size, data_test_size = get_data_ready(tensor_size=(28, 28))

    set_random_seeds(seed_id=train_config.seed_number, verbose=False)
    net = VAE(d=train_config.latent_code_size)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    # Get first stats for untrained network
    train_loss = test(net, data_train_loader, train_config.vae_beta)
    test_loss = test(net, data_test_loader, train_config.vae_beta)

    time_tic = [0]
    stats_tic = [train_loss, test_loss]
    train_log.update_log(time_tic, stats_tic, model=net, save=True)

    # Loop over training epochs
    for ep in range(train_config.num_epochs):
        net, train_loss = train(net, optimizer, data_train_loader,
                                train_config.vae_beta)
        test_loss = test(net, data_test_loader,
                         train_config.vae_beta)
        time_tic = [ep+1]
        stats_tic = [train_loss, test_loss]
        train_log.update_log(time_tic, stats_tic, model=net, save=True)


def train(net, optimizer, train_loader, beta):
    net.train()
    train_loss = 0
    for x, _ in train_loader:
        x = x
        # ===================forward=====================
        x_hat, mu, logvar = net(x)
        loss = vae_loss(x_hat, x, mu, logvar, beta)
        train_loss += loss.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    train_loss /= len(train_loader.dataset)
    return net, train_loss


def test(net, test_loader, beta):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        for x, y in test_loader:
            x = x
            # ===================forward=====================
            x_hat, mu, logvar = net(x)
            test_loss += vae_loss(x_hat, x, mu, logvar, beta).item()
    # ===================log========================
    test_loss /= len(test_loader.dataset)
    return test_loss


if __name__ == '__main__':
    train_config, net_config, log_config = get_configs_ready(
        default_config_fname="configs/train/mnist_vae.json")
    main(net_config, train_config, log_config)
