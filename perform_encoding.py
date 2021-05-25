# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from encoding_models.ols import OLS_pytorch
from utils.helper import save_dict, load_dict
from utils.evaluate import vectorized_correlation


def get_activations(activations_dir, layer_name):
    """ Loads NN features into a np array according to layer. """
    train_file = os.path.join(activations_dir,"train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir,"test_" + layer_name + ".npy")
    train_activations = np.load(train_file)
    test_activations = np.load(test_file)
    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)
    return train_activations, test_activations


def get_fmri(fmri_dir, ROI):
    """ Loads fMRI data into a numpy array for to a given ROI.
    matrix of dimensions #train_vids x #repetitions x #voxels
    containing fMRI responses to train videos of a given ROI
    """
    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions? WHAT IS MEANED HERE? TIME?
    ROI_data_train = np.mean(ROI_data["train"], axis = 1)
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train


def predict_fmri_fast(train_activations, test_activations,
                      train_fmri, use_gpu=False):
    """
    Parameters
    ----------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos.
    train_fmri : np.array
        matrix of dimensions #train_vids x  #voxels
        containing fMRI responses to train videos
    Returns
    -------
    fmri_pred_test: np.array
        matrix of dimensions #test_vids x  #voxels
        containing predicted fMRI responses to test videos .
    """
    reg = OLS_pytorch(lambda_reg=0.01, use_gpu=use_gpu)
    reg.fit(train_activations, train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)
    return fmri_pred_test


def main(sub='sub04', ROI='EBA', layer='layer_5'):
    mode = 'val'
    model = 'alexnet_devkit'
    batch_size = 1000
    activation_dir = './data/features/alexnet/pca_50'
    result_dir = './data/results'
    fmri_dir = './data/participants_data_v2021'

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"

    fmri_dir = os.path.join(fmri_dir, track)

    sub_fmri_dir = os.path.join(fmri_dir, sub)
    results_dir = os.path.join(result_dir, model, layer, track, sub)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_activations,test_activations = get_activations(activation_dir, layer)
    if track == "full_track":
        fmri_train_all,voxel_mask = get_fmri(sub_fmri_dir,ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir,ROI)
    num_voxels = fmri_train_all.shape[1]
    if mode == 'val':
        # Here as an example we use first 900 videos as training and rest of the videos as validation
        test_activations = train_activations[900:,:]
        train_activations = train_activations[:900,:]
        fmri_train = fmri_train_all[:900,:]
        fmri_test = fmri_train_all[900:,:]
        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_val.npy')
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos,num_voxels))
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_test.npy')

    iter = 0
    while iter < num_voxels-batch_size:
        pred_fmri[:,iter:iter+batch_size] = predict_fmri_fast(train_activations,test_activations,
                                                              fmri_train[:,iter:iter+batch_size],
                                                              use_gpu = use_gpu)
        iter = iter+batch_size
        #print((100*iter)//num_voxels," percent complete")
    pred_fmri[:,iter:] = predict_fmri_fast(train_activations, test_activations,
                                           fmri_train[:,iter:iter+batch_size],
                                           use_gpu = use_gpu)

    if mode == 'val':
        score = vectorized_correlation(fmri_test,pred_fmri)
        # print(f"Subject: {sub} | ROI: {ROI} | Layer: {layer} | # voxels {num_voxels} | Corr: {round(score.mean(), 3)}")

    np.save(pred_fmri_save_path, pred_fmri)
    return round(score.mean(), 3), num_voxels

if __name__ == "__main__":
    all_subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
                    'sub06', 'sub07', 'sub08', 'sub09', 'sub10']
    all_layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4',
                  'layer_5', 'layer_6', 'layer_7', 'layer_8']
    all_rois = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']
    for ROI in all_rois:
        for sub in all_subjects:
            all_scores = []
            for layer in all_layers:
                score_l, num_voxels = main(sub, ROI, layer)
                all_scores.append(score_l)
            best_layer_id = np.argmax(all_scores)
            print(f"Subject: {sub} | ROI: {ROI} | Best Layer: {all_layers[best_layer_id]} | # voxels {num_voxels} | Corr: {all_scores[best_layer_id]}")
        print("----------------------------------------------------------------------------")
