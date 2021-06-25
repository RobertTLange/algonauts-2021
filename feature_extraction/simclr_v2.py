import torch
from simclr_v2_model.resnet import get_simclr_resnet, name_to_params


model_cktp = {
    'simclr_r50_1x_sk0_100pct': 'simclr_v2_model/torch_ckpt/r50_1x_sk0_finetuned_100pct.pth',
    'simclr_r50_2x_sk1_100pct': 'simclr_v2_model/torch_ckpt/r50_2x_sk1_finetuned_100pct.pth',
    'simclr_r101_1x_sk0_100pct': 'simclr_v2_model/torch_ckpt/r101_1x_sk0_finetuned_100pct.pth',
    'simclr_r101_1x_sk1_100pct': 'simclr_v2_model/torch_ckpt/r101_1x_sk1_finetuned_100pct.pth',
    'simclr_r101_2x_sk0_100pct': 'simclr_v2_model/torch_ckpt/r101_2x_sk0_finetuned_100pct.pth',
    'simclr_r101_2x_sk1_100pct': 'simclr_v2_model/torch_ckpt/r101_2x_sk1_finetuned_100pct.pth',
    'simclr_r152_2x_sk1_100pct': 'simclr_v2_model/torch_ckpt/r152_2x_sk1_finetuned_100pct.pth',
    'simclr_r152_3x_sk1_100pct': 'simclr_v2_model/torch_ckpt/r152_3x_sk1_finetuned_100pct.pth',
}

all_simclr_models = ['r50_1x_sk0', 'r50_2x_sk1',
                     'r101_1x_sk0', 'r101_1x_sk1',
                     'r101_2x_sk0', 'r101_2x_sk1',
                     'r152_2x_sk1', 'r152_3x_sk1']

def load_simclr_v2(simclr_type):
    assert simclr_type in model_cktp.keys()
    model, _ = get_simclr_resnet(*name_to_params(model_cktp[simclr_type]))
    model.load_state_dict(torch.load(model_cktp[simclr_type])['resnet'])

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
