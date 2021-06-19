import torch
import timm


def load_timm(model_name='resnest26d', features_only=True):
    all_pretrained_nets = timm.list_models(pretrained=True)
    assert model_name in all_pretrained_nets, "Not in pretrained timm models."
    m = timm.create_model(model_name, features_only=features_only, pretrained=True)
    if torch.cuda.is_available():
        m.cuda()
    m.eval()
    return m


if __name__ == '__main__':
    load_timm_model(model_name='resnest26d')
