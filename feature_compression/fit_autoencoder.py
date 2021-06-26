from .autoencoder import Autoencoder
from .train_autoencoder import fit_autoencoder
import torch

def fit_trafo_autoencoder(x_train, x_test, ae_params):
    net = Autoencoder(input_dim=x_train.shape[1],
                      d=ae_params["n_components"])
    net, time, stats = fit_autoencoder(net, x_train,
                                       batch_size=128,
                                       num_epochs=50)
    # trafo = dim_red.fit(x_train)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    x_train = torch.Tensor(x_train).to(device).float()
    x_test = torch.Tensor(x_test).to(device).float()
    #print(x_train.shape, x_test.shape)
    x_train_trafo, _ = net(x_train)
    x_test_trafo, _ = net(x_test)
    #print(x_train_trafo.shape, x_test_trafo.shape)
    ae_info = {"time": time, "stats": stats}
    return (x_train_trafo.detach().numpy(),
            x_test_trafo.detach().numpy(), ae_info)
