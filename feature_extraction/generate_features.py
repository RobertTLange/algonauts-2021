import os
import glob
from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA

from alexnet import load_alexnet
from vgg import load_vgg
from resnet import load_resnet
from timm_models import load_timm


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
                             sampling_rate = 4):
    """ how many frames to skip when feeding into the network. """
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for video_file in tqdm(video_list):
        vid,num_frames = get_video_from_mp4(video_file, sampling_rate)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = []
        for frame in range(num_frames):
            img =  vid[0,frame,:,:,:]
            input_img = V(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img=input_img.cuda()
            x = model.forward(input_img)
            for i, feat in enumerate(x):
                if frame==0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] =  activations[i] + feat.data.cpu().numpy().ravel()
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir,
                                     video_file_name + "_" + "layer"
                                     + "_" + str(layer+1) + ".npy")
            np.save(save_path, activations[layer]/float(num_frames))
    return len(activations)


def do_PCA_and_save(activations_dir, save_dir, num_layers, num_pca_dims):
    # TODO: Layerwise vs aggregated PCA
    layers = ['layer_' + str(l+1) for l in range(num_layers)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir +'/*'+layer+'.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0]).shape[0]
        x = np.zeros((len(activations_file_list),feature_dim))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i,:] = temp
        x_train = x[:1000,:]
        x_test = x[1000:]
        #print(x.shape, x_train.shape, x_test.shape)
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        # Full vs incremental (depending on pca dim and sampling rate)
        pca = PCA(n_components=num_pca_dims)#, batch_size=20)
        pca.fit(x_train)
        #print(pca.explained_variance_)
        #print(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_.cumsum()[-1])

        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        #print(x.shape, x_train.shape, x_test.shape)
        train_save_path = os.path.join(save_dir, "train_" + layer)
        test_save_path = os.path.join(save_dir, "test_" + layer)
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)


def main(model_type, pca_dims, save_dir, video_dir):
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
    else:
        model = load_timm(model_type)
    print(f'{model_type} Model loaded')

    # get and save activations from raw video
    activations_dir = os.path.join(save_dir, "activations")
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    num_layers = get_activations_and_save(model, video_list, activations_dir)

    # preprocessing using PCA and save
    for num_pca_dims in pca_dims:
        pca_dir = os.path.join(save_dir, f'pca_{num_pca_dims}')
        print(f"------performing  PCA: {num_pca_dims}---------")
        do_PCA_and_save(activations_dir, pca_dir, num_layers, num_pca_dims)


if __name__ == "__main__":
    video_dir = './data/AlgonautsVideos268_All_30fpsmax/'
    pca_dims = [50, 100, 250, 500, 1000]
    all_models = [
                  #'alexnet',
                  #'vgg',
                  #'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  #'efficientnet_b3', 'resnext50_32x4d'
                  ]
    # Loop over all models, create features from forward passes and reduce dims
    for model_type in all_models:
        save_dir = f'./data/features/{model_type}'
        main(model_type, pca_dims, save_dir, video_dir)
