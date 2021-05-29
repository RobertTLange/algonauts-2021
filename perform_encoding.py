# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from utils.helper import save_dict, load_dict, get_activations, get_fmri
from utils.evaluate import vectorized_correlation


def perform_test_encoding(sub='sub04', ROI='EBA', layer='layer_5', mode='val'):
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
    results_dir = os.path.join(result_dir, model, track, sub)
    if mode == "test":
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    train_activations,test_activations = get_activations(activation_dir, layer)
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir,ROI)
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
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos, num_voxels))
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_test.npy')

    iter = 0
    while iter < num_voxels-batch_size:
        pred_fmri[:,iter:iter+batch_size] = predict_fmri_fast(train_activations,
                                                              test_activations,
                                                              fmri_train[:,iter:iter+batch_size],
                                                              use_gpu=use_gpu)
        iter = iter+batch_size
        #print((100*iter)//num_voxels," percent complete")
    pred_fmri[:,iter:] = predict_fmri_fast(train_activations, test_activations,
                                           fmri_train[:,iter:iter+batch_size],
                                           use_gpu = use_gpu)

    if mode == 'val':
        score = vectorized_correlation(fmri_test,pred_fmri)
        # print(f"Subject: {sub} | ROI: {ROI} | Layer: {layer} | # voxels {num_voxels} | Corr: {round(score.mean(), 3)}")
        return round(score.mean(), 3), num_voxels
    else:
        np.save(pred_fmri_save_path, pred_fmri)
        return pred_fmri_save_path


if __name__ == "__main__":
    # Generates submission for best layer per subject/roi evaluated on train/test split
    # Results stored as results/alexnet_devkit/mini_track/<subject_id>/<ROI>_test.npy
    all_subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
                    'sub06', 'sub07', 'sub08', 'sub09', 'sub10']
    all_layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4',
                  'layer_5', 'layer_6', 'layer_7', 'layer_8']
    all_rois = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

    for ROI in all_rois:
        subject_scores = []
        for sub in all_subjects:
            all_scores = []
            for layer in all_layers:
                score_l, num_voxels = main(sub, ROI, layer, mode="val")
                all_scores.append(score_l)
            best_layer_id = np.argmax(all_scores)
            subject_scores.append(all_scores[best_layer_id])
            print(f"Subject: {sub} | ROI: {ROI} | Best Layer: {all_layers[best_layer_id]} | # voxels {num_voxels} | Corr: {all_scores[best_layer_id]}")
            pred_fmri_save_path = main(sub, ROI, all_layers[best_layer_id], mode="test")
            print(f"Stored in {pred_fmri_save_path}")
        print(f"ROI: {ROI} | Mean Corr: {np.mean(subject_scores)}")
        print("----------------------------------------------------------------------------")
