import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50 as _resnet50


model_ckpt = {
    'barlow_twins_1p_20ep': 'feature_extraction/barlowtwins_model/checkpoint/semisup/checkpoint.pth',
    'barlow_twins_10p_20ep': 'feature_extraction/barlowtwins_model/checkpoint/semisup10/checkpoint.pth',
    'barlow_twins_10p_50ep': 'feature_extraction/barlowtwins_model/checkpoint/semisup10_ep50/checkpoint.pth',
    'barlow_twins_no_tuning': 'https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth'
}


def load_barlow_twins(model_type):
    assert model_type in model_ckpt.keys()
    resnet = _resnet50(pretrained=False)
    ckpt = model_ckpt[model_type]

    if model_type == "barlow_twins_no_tuning":
        state_dict = torch.hub.load_state_dict_from_url(ckpt, map_location='cpu')
    else:
        state_dict = torch.load(ckpt)
    resnet.load_state_dict(state_dict, strict=False)
    #model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    model = BarlowResNet(resnet)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class BarlowResNet(nn.Module):
    def __init__(self, original_model):
        super(BarlowResNet, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.resnet = original_model

    def forward(self, x):
        all_out = []
        x = self.resnet.conv1(x)
        all_out.append(x)
        x = self.resnet.bn1(x)
        all_out.append(x)
        x = self.resnet.relu(x)
        all_out.append(x)
        x = self.resnet.maxpool(x)
        all_out.append(x)

        for l in self.resnet.layer1:
            x = l(x)
            all_out.append(x)

        for l in self.resnet.layer2:
            x = l(x)
            all_out.append(x)

        for l in self.resnet.layer3:
            x = l(x)
            all_out.append(x)

        for l in self.resnet.layer4:
            x = l(x)
            all_out.append(x)

        x = self.resnet.avgpool(x)
        all_out.append(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        all_out.append(x)

        return all_out
