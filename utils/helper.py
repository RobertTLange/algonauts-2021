import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
        #print(p)
        #ret_di = pickle.load(f)
    return ret_di


def saveasnii(brain_mask, nii_save_path, nii_data):
    import nibabel as nib
    img = nib.load(brain_mask)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)


def get_encoding_data(fmri_dir='./data/participants_data_v2021',
                      activations_dir='./data/features/alexnet/pca_50',
                      layer_id='layer_1', subject_id='sub01', roi_type='V1'):
    if roi_type == "WB": track = "full_track"
    else: track = "mini_track"
    X, X_test = get_activations(activations_dir, layer_id)
    y = get_fmri(fmri_dir, track, subject_id, roi_type)
    return X, y, X_test


def get_activations(activations_dir, layer_name):
    """ Loads NN features into a np array according to layer. """
    train_file = os.path.join(activations_dir, "train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir, "test_" + layer_name + ".npy")
    train_activations = np.load(train_file)
    test_activations = np.load(test_file)
    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)
    return train_activations, test_activations


def get_fmri(fmri_dir, track, subject_id, ROI, mean=True):
    """ Loads fMRI data into a numpy array for to a given ROI.
    matrix of dimensions #train_vids x #repetitions x #voxels
    containing fMRI responses to train videos of a given ROI
    """
    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, track, subject_id, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions? WHAT IS MEANED HERE? TIME?
    if mean:
        ROI_data_train = np.mean(ROI_data["train"], axis = 1)
    else:
        ROI_data_train = ROI_data["train"]
    return ROI_data_train
