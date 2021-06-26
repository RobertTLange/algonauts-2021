import os
import glob
from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn

from .alexnet import load_alexnet
from .vgg import load_vgg
from .resnet import load_resnet
from .timm_models import load_timm
from .vonenet import load_vonenet
from .simclr_v2 import load_simclr_v2
from sklearn.decomposition import PCA
import scipy.stats as stats


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
        (ret, frame) = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if fc % sampling_rate == 0:
            buf[int((fc - 1) / sampling_rate)] = frame

    cap.release()
    return np.expand_dims(buf, axis=0),int(frameCount / sampling_rate)


def get_activations_and_save(model, video_list, activations_dir,
                             model_type,
                             filter_config={"filter_name": "mean",
                                            "sampling_rate": 2}):
    """
    how many frames to skip when feeding into the network.
    sampling rate = 1 -> 70-91 frames
    sampling rate = 2 -> 45 frames
    sampling rate = 3 -> 30 frames
    sampling rate = 4 -> 11-22 frames
    """
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
        vid, num_frames = get_video_from_mp4(video_file,
                                             filter_config["sampling_rate"])
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
                    activations.append([feat.data.cpu().numpy().ravel()])
                else:
                    activations[i].append(feat.data.cpu().numpy().ravel())

        # Features are meaned!
        for layer in range(len(activations)):
            stacked_activations =  np.stack(activations[layer], axis=1)
            #print(stacked_activations.shape)
            save_path = os.path.join(activations_dir,
                                     video_file_name +
                                     "_layer_" + str(layer+1) + ".npy")

            # Different Temporal Smoothing Methods
            if filter_config["filter_name"] == "raw":
                filtered_activations = stacked_activations
            elif filter_config["filter_name"] == "mean":
                filtered_activations = np.mean(stacked_activations, axis=1)
            elif filter_config["filter_name"] == "1d-pca":
                filtered_activations = temporal_activation_pca(stacked_activations)
            elif filter_config["filter_name"].startswith("bold-kernel"):
                filtered_activations = temporal_bold_kernel(stacked_activations,
                                                            filter_config["peak"],
                                                            filter_config["under"],
                                                            filter_config["under_coeff"])
            np.save(save_path, filtered_activations)
        #     break
        # break
    return len(activations)


def temporal_activation_pca(stacked_activations):
    pca = PCA(n_components=1)
    pca.fit(stacked_activations)
    x_train_trafo = pca.transform(stacked_activations)
    return x_train_trafo


def hrf(times, peak=20, under=40, under_coeff=0.35):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = stats.gamma.pdf(times, peak)
    # Gamma pdf for the undershoot
    undershoot_values = stats.gamma.pdf(times, under)
    # Combine them
    values = peak_values - under_coeff * undershoot_values
    # Scale max to 0.6
    return values / np.sum(values)


def temporal_bold_kernel(stacked_activations, peak, under, under_coeff):
    num_neurons, num_frames = stacked_activations.shape
    times = np.linspace(1, 100, num_frames)
    hrf_kernel = hrf(times, peak, under, under_coeff)
    convolved_activations = []
    for neuron_id in range(num_neurons):
        conv_neuron = np.convolve(stacked_activations[neuron_id],
                                  hrf_kernel)[:-(num_frames - 1)]
        convolved_activations.append(conv_neuron)
    stacked_convolved = np.stack(convolved_activations, axis=0)
    return np.mean(stacked_activations, axis=1)


def run_activation_features(model_type, save_dir, video_dir, filter_config):
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
    elif model_type.startswith('simclr'):
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
                                          model_type, filter_config)



if __name__ == "__main__":
    video_dir = '../data/AlgonautsVideos268_All_30fpsmax/'
    all_models = [
                  # 'alexnet',
                  # 'vgg',
                  # 'resnet18', 'resnet34',
                  # 'resnet50',
                  # 'resnet101', 'resnet152',
                  # 'efficientnet_b3', 'resnext50_32x4d',
                  # "vone-resnet50",
                  # "vone-resnet50_at",
                  # "vone-resnet50_ns",
                  # "vone-cornets",
                  # 'simclr_r50_1x_sk0_100pct',
                  # 'simclr_r50_2x_sk1_100pct',
                  'simclr_r101_1x_sk0_100pct',
                  'simclr_r101_1x_sk1_100pct',
                  'simclr_r101_2x_sk0_100pct',
                  'simclr_r101_2x_sk1_100pct',
                  'simclr_r152_2x_sk1_100pct',
                  'simclr_r152_3x_sk1_100pct'
                  ]
    filter_configs = [
                     # {"filter_name": "raw",
                     # "sampling_rate": 4},
                     {"filter_name": "mean",
                      "sampling_rate": 4}
                     # {"filter_name": "1d-pca",
                     #  "sampling_rate": 4},
                     # {"filter_name": "bold-kernel-1",
                     #  "sampling_rate": 4,
                     #  "peak": 20,
                     #  "under": 40,
                     #  'under_coeff': 0.35},
                     # {"filter_name": "bold-kernel-2",
                     #  "sampling_rate": 4,
                     #  "peak": 60,
                     #  "under": 80,
                     #  'under_coeff': 0.5},
                     # {"filter_name": "bold-kernel-3",
                     #  "sampling_rate": 4,
                     #  "peak": 40,
                     #  "under": 60,
                     #  'under_coeff': 0.35}
                     ]

    # Loop over all models, create features from forward passes and reduce dims
    for filter_config in filter_configs:
        for model_type in all_models:
            save_dir = f'../data/features/{model_type}/{filter_config["filter_name"]}'
            print(save_dir)
            run_activation_features(model_type, save_dir,
                                    video_dir, filter_config)
