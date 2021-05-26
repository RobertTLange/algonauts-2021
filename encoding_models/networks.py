import torch
import torch.nn as nn
from encoding_models.body_builder import BodyBuilder
from encoding_models.train_dojo import TrainDojo


def fit_mlp_model(model_config, X, y):
    net_config = {"input_dim": (1, 2000),
                  "layers_info": [["flatten"],
                                  ["linear", 512, True],
                                  ["linear", 100, True]],
                  "output_act": "identity",
                  "hidden_act": "relu",
                  "dropout": 0.0,
                  "batch_norm": False}
    model = BodyBuilder(**net_config)
    return


mlp_params_to_search = {}
