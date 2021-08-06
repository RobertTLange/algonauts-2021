import os
import glob
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from dotmap import DotMap
import zipfile


def get_best_roi_models(json_path, prepend=None):
    with open(json_path) as json_file:
        expert_ckpt = DotMap(json.load(json_file))
    if prepend is not None:
        for k in expert_ckpt.keys():
            expert_ckpt[k]["path"] = os.path.join(prepend, expert_ckpt[k]["path"])
    return expert_ckpt


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
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

    # Loop over layers and stack features together
    if type(layer_id) == list:
        all_X = []
        all_X_test = []
        for l_id in layer_id:
            X_t, X_t_test = get_activations(activations_dir, l_id)
            all_X.append(X_t)
            all_X_test.append(X_t_test)
        X, X_test = np.concatenate(all_X, axis=1), np.concatenate(all_X_test, axis=1)
    else:
        X, X_test = get_activations(activations_dir, layer_id)
    y = get_fmri(fmri_dir, subject_id, roi_type)
    return X, y, X_test


def get_activations(activations_dir, layer_name):
    """ Loads NN features into a np array according to layer. """
    # Use transformed features
    if not activations_dir.endswith('activations'):
        train_file = os.path.join(activations_dir, "train_" + layer_name + ".npy")
        test_file = os.path.join(activations_dir, "test_" + layer_name + ".npy")
        train_activations = np.load(train_file)
        test_activations = np.load(test_file)
    # Use raw activations (e.g. together with PLS)
    else:
        activations_file_list = glob.glob(activations_dir +'/*' + layer_name + '.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0]).shape[0]
        x = np.zeros((len(activations_file_list), feature_dim))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i, :] = temp
        train_activations = x[:1000]
        test_activations = x[1000:]
    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)
    return train_activations, test_activations


def get_fmri(fmri_dir, subject_id, roi_type, mean=True):
    """ Loads fMRI data into a numpy array for to a given ROI.
    matrix of dimensions #train_vids x #repetitions x #voxels
    containing fMRI responses to train videos of a given ROI
    """
    if roi_type == "WB": track = "full_track"
    else: track = "mini_track"
    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, track, subject_id, roi_type + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions? WHAT IS MEANED HERE? TIME?
    if mean:
        ROI_data_train = np.mean(ROI_data["train"], axis = 1)
    else:
        ROI_data_train = ROI_data["train"]
    return ROI_data_train


def zip(src, dst):
    ''' Zip Submission Results'''
    zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            if arcname in ["mini_track.pkl", "full_track.pkl"]:
                zf.write(absname, arcname)
    zf.close()
