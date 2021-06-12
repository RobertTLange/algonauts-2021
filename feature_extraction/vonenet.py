import torch
import torch.nn as nn
import torchvision
from vonenet_model import get_model

model_archs = ['alexnet', 'resnet50', 'resnet50_at',
               'resnet50_ns', 'cornets']


def load_vonenet(vonenet_type='resnet50'):
    assert vonenet_type in model_archs
    model = get_model(model_arch=vonenet_type,
                      pretrained=True)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    model = load_vonenet()
    x = torch.empty((1, 3, 224, 224)).normal_(mean=0, std=1)
    feature_list = []
    # Iterate over layers and collect intermediate feature representations
    for layer in model:
        x = layer(x)
        if type(x) == tuple:
            feature_list.extend(x)
        else:
            feature_list.append(x)

    for i  in feature_list:
        print(i.shape)
