import torch
from simclr_v2.resnet import get_simclr_resnet, name_to_params


model_cktp = {
    'simclr_r50_1x_sk0_100pct': 'simclr_v2/torch_ckpt/r50_1x_sk0_finetuned_100pct.pth',
    'simclr_r50_1x_sk0_10pct': 'simclr_v2/torch_ckpt/r50_1x_sk0_finetuned_10pct.pth',
    'simclr_r50_1x_sk0_1pct': 'simclr_v2/torch_ckpt/r50_1x_sk0_finetuned_1pct.pth',
    'simclr_r50_2x_sk1_100pct': 'simclr_v2/torch_ckpt/r50_2x_sk1_finetuned_100pct.pth',
    'simclr_r50_2x_sk1_10pct': 'simclr_v2/torch_ckpt/r50_2x_sk1_finetuned_10pct.pth',
    'simclr_r50_2x_sk1_1pct': 'simclr_v2/torch_ckpt/r50_2x_sk1_finetuned_1pct.pth',
    'simclr_r150_3x_sk1_100pct': 'simclr_v2/torch_ckpt/r150_3x_sk1_finetuned_100pct.pth',
    'simclr_r150_3x_sk1_10pct': 'simclr_v2/torch_ckpt/r150_3x_sk1_finetuned_10pct.pth',
    'simclr_r150_3x_sk1_1pct': 'simclr_v2/torch_ckpt/r150_3x_sk1_finetuned_1pct.pth',
}


def load_simclr_v2(simclr_type):
    assert simclr_type in model_cktp.keys()
    model, _ = get_simclr_resnet(*name_to_params(model_cktp[simclr_type]))
    model.load_state_dict(torch.load(pth_path)['resnet'])

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
