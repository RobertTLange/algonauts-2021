import os
import glob
from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn

from alexnet import load_alexnet
from vgg import load_vgg
from resnet import load_resnet
from timm_models import load_timm
from vonenet import load_vonenet
from simclr_v2 import load_simclr_v2


def get_video_from_mp4(file, sampling_rate):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((int(frameCount / sampling_rate), frameHeight,
                   frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        fc += 1
        if fc % sampling_rate == 0:
            (ret, buf[int((fc - 1) / sampling_rate)]) = cap.read()

    cap.release()
    return np.expand_dims(buf, axis=0),int(frameCount / sampling_rate)


def get_activations_and_save(model, video_list, activations_dir,
                             model_type, sampling_rate=4):
    """ how many frames to skip when feeding into the network. """
    if model_type in ["vone-alexnet", "vone-resnet50",
                      "vone-resnet50_at", "vone-resnet50_ns",
                      "vone-cornets"]:
        centre_crop = trn.Compose([
                trn.ToPILImage(),
                trn.Resize((224,224)),
                trn.ToTensor(),
                trn.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5])])
    elif model_type in ['simclr_r50_1x_sk0_100pct',
                        'simclr_r50_1x_sk0_10pct',
                        'simclr_r50_1x_sk0_1pct',
                        'simclr_r50_2x_sk1_100pct',
                        'simclr_r50_2x_sk1_10pct',
                        'simclr_r50_2x_sk1_1pct',
                        'simclr_r152_3x_sk1_100pct',
                        'simclr_r152_3x_sk1_10pct',
                        'simclr_r152_3x_sk1_1pct']:
        centre_crop = trn.Compose([
                trn.ToPILImage(),
                trn.Resize((224,224)),
                trn.ToTensor()])
    else:
        centre_crop = trn.Compose([
                trn.ToPILImage(),
                trn.Resize((224,224)),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

    for video_file in tqdm(video_list):
        vid,num_frames = get_video_from_mp4(video_file, sampling_rate)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = []
        for frame in range(num_frames):
            img =  vid[0,frame,:,:,:]
            input_img = V(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.cuda()

            if model_type in ["vone-alexnet", "vone-resnet50",
                              "vone-resnet50_at", "vone-resnet50_ns",
                              "vone-cornets"]:
                x = []
                for layer in model:
                    input_img = layer(input_img)
                    if type(input_img) == tuple:
                        x.extend(input_img)
                    else:
                        x.append(input_img)
            else:
                x = model.forward(input_img)

            for i, feat in enumerate(x):
                if frame == 0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] =  activations[i] + feat.data.cpu().numpy().ravel()
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir,
                                     video_file_name + "_" + "layer"
                                     + "_" + str(layer+1) + ".npy")
            np.save(save_path, activations[layer]/float(num_frames))
    return len(activations)


def run_activation_features(model_type, save_dir, video_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load Alexnet
    if model_type == "alexnet":
        model = load_alexnet()
    elif model_type in ['resnet18', 'resnet34', 'resnet50',
                        'resnet101', 'resnet152']:
        model = load_resnet(model_type)
    elif model_type == "vgg":
        model = load_vgg()
    elif model_type in ["vone-alexnet",
                        "vone-resnet50",
                        "vone-resnet50_at",
                        "vone-resnet50_ns",
                        "vone-cornets"]:
        model_name = model_type.split("-")[1]
        model = load_vonenet(model_name)
    elif model_type in ['simclr_r50_1x_sk0_100pct',
                        'simclr_r50_1x_sk0_10pct',
                        'simclr_r50_1x_sk0_1pct',
                        'simclr_r50_2x_sk1_100pct',
                        'simclr_r50_2x_sk1_10pct',
                        'simclr_r50_2x_sk1_1pct',
                        'simclr_r152_3x_sk1_100pct',
                        'simclr_r152_3x_sk1_10pct',
                        'simclr_r152_3x_sk1_1pct']:
        model = load_simclr_v2(model_type)
    else:
        model = load_timm(model_type)
    print(f'{model_type} Model loaded')

    # get and save activations from raw video
    activations_dir = os.path.join(save_dir, "activations")
    print(activations_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    num_layers = get_activations_and_save(model, video_list, activations_dir,
                                          model_type)



if __name__ == "__main__":
    video_dir = '../data/AlgonautsVideos268_All_30fpsmax/'
    all_models = [
                  #'alexnet',
                  #'vgg',
                  #'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  #'efficientnet_b3', 'resnext50_32x4d',
                  #"vone-resnet50",
                  #"vone-resnet50_at",
                  #"vone-resnet50_ns",
                  #"vone-cornets",
                  'simclr_r50_1x_sk0_100pct', 'simclr_r50_1x_sk0_10pct', 'simclr_r50_1x_sk0_1pct',
                  'simclr_r50_2x_sk1_100pct', 'simclr_r50_2x_sk1_10pct', 'simclr_r50_2x_sk1_1pct',
                  # 'simclr_r152_3x_sk1_100pct', 'simclr_r152_3x_sk1_10pct', 'simclr_r152_3x_sk1_1pct'
                  ]

    # Loop over all models, create features from forward passes and reduce dims
    for model_type in all_models:
        save_dir = f'../data/features/{model_type}'
        run_activation_features(model_type, save_dir, video_dir)
