import torch
import timm


def load_timm_model(model_name='resnest26d'):
    all_pretrained_nets = timm.list_models(pretrained=True)
    assert model_name in all_pretrained_nets, "Not in pretrained timm models."
    m = timm.create_model('resnest26d', features_only=True, pretrained=True)
    m.eval()
    return m


if __name__ == '__main__':
    load_timm_model(model_name='resnest26d')
