import torch
import torch.nn as nn
from encoding_models.networks.body_builder import BodyBuilder
from encoding_models.networks.train_dojo import TrainDojo
from encoding_models.networks.data_loader import get_train_test_split_loaders
from encoding_models.networks.optimizer import set_optimizer

mlp_params_to_search = {
"categorical":
    {"num_hidden_layers": [1, 2, 3, 4],
     "num_hidden_units": [64, 128, 256, 512],
     "hidden_act": ["relu", "prelu", "leaky_relu", "elu"],
     "optimizer_class": ["Adam", "RMSprop"],
     "learning_rate": [1e-05, 5e-05, 1e-04, 5e-04, 1e-03],
     "w_decay": [0.0, 1e-07, 1e-06, 1e-05],
     "dropout": [0.0, 0.1, 0.2]}
}


mlp_default_params = {"batch_size": 32,
                      "num_epochs": 20,
                      "torch_num_threads": 5,
                      "early_stop_patience": 7,
                      "early_stop_val_split": 0.1}


def fit_mlp_model(model_config, X, y):
    """ Build and train MLP Encoder with early stopping. """
    # Build model config based on proposed model_config
    core_layers = (int(model_config["num_hidden_layers"])
                   *[["linear", int(model_config["num_hidden_units"]), True]])
    layers_info = [["flatten"],
                    *core_layers,
                    ["linear", y.shape[1], True]]
    net_config = {"input_dim": (1, X.shape[1]),
                  "layers_info": layers_info,
                  "output_act": "identity",
                  "hidden_act": model_config["hidden_act"],
                  "dropout": model_config["dropout"],
                  "batch_norm": False}

    # Create model, optimizer and criterion
    network = BodyBuilder(**net_config)
    optimizer = set_optimizer(network,
                              model_config["optimizer_class"],
                              model_config["learning_rate"],
                              model_config["w_decay"])
    criterion = nn.MSELoss()

    # Use 10% of cross-val train data for early stopping criterion!
    train_loader, val_loader = get_train_test_split_loaders(X, y,
                                mlp_default_params["early_stop_val_split"],
                                mlp_default_params["batch_size"])
    torch.set_num_threads(mlp_default_params["torch_num_threads"])
    dojo = TrainDojo(network, optimizer, criterion,
                     train_loader, val_loader,
                     patience=mlp_default_params["early_stop_patience"],
                     log_batch_interval=10)
    dojo.train(20)
    return network


def predict_mlp_model(model_params, X):
    data = torch.Tensor(X).to(torch.device('cpu'))
    return model_params(data.float()).detach().numpy()
