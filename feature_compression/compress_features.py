import os
import glob
from tqdm import tqdm
import numpy as np
from .fit_pca import fit_trafo_pca
from .fit_umap import fit_trafo_umap
from .fit_autoencoder import fit_trafo_autoencoder
from .fit_mds import fit_trafo_mds

from mle_toolbox.utils import save_pkl_object
from sklearn.preprocessing import StandardScaler


def do_dim_reduction_and_save(activations_dir, save_dir,
                              trafo_type, dim_red_params,
                              info_title):
    onlyfiles = [f for f in os.listdir(activations_dir)
                 if os.path.isfile(os.path.join(activations_dir, f))]
    num_layers = int(len(onlyfiles)/1102)
    print("Number of layers", num_layers)

    layers = ['layer_' + str(l+1) for l in range(num_layers)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        explained_variance = []

    all_layers_info = {}
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir +'/*'+layer+'.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0]).shape[0]
        x = np.zeros((len(activations_file_list),feature_dim))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i,:] = temp.squeeze()
        x_train = x[:1000,:]
        x_test = x[1000:]
        #print(x.shape, x_train.shape, x_test.shape)S
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)

        # Plug&Play - Fitting of dimensionality reduction technique
        if trafo_type == "pca":
            x_train_trafo, x_test_trafo, info_l = fit_trafo_pca(x_train, x_test,
                                                                dim_red_params)
        elif trafo_type == "umap":
            x_train_trafo, x_test_trafo, info_l = fit_trafo_umap(x_train, x_test,
                                                                 dim_red_params)
        elif trafo_type == "autoencoder":
            x_train_trafo, x_test_trafo, info_l = fit_trafo_autoencoder(
                                                               x_train, x_test,
                                                               dim_red_params)
        elif trafo_type == "mds":
            x_train_trafo, x_test_trafo, info_l = fit_trafo_mds(x_train, x_test,
                                                                dim_red_params)
        #print(x_train.shape, x_train_trafo.shape)
        train_save_path = os.path.join(save_dir, "train_" + layer)
        test_save_path = os.path.join(save_dir, "test_" + layer)
        np.save(train_save_path, x_train_trafo)
        np.save(test_save_path, x_test_trafo)

        all_layers_info[layer] = info_l
    info_path = os.path.join(save_dir, info_title)
    save_pkl_object(all_layers_info, info_path)
    print(all_layers_info)


def run_compression(save_dir, model_type, trafo_type, num_components):
    activations_dir = os.path.join(save_dir, "activations")
    # preprocessing using PCA and save
    info_title = f'{model_type}_{trafo_type}_{num_components}.pkl'
    dim_red_dir = os.path.join(save_dir, f'{trafo_type}_{num_components}')
    print(dim_red_dir)
    print(f"------performing  {trafo_type}: {num_components}---------")
    dim_red_params = {"n_components": num_components}
    do_dim_reduction_and_save(activations_dir,
                              dim_red_dir,
                              trafo_type,
                              dim_red_params,
                              info_title)


if __name__ == "__main__":
    all_models = [
                  # 'alexnet',
                  # 'vgg',
                  # 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  # 'efficientnet_b3', 'resnext50_32x4d',
                  # #"vone-alexnet",
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

    # Loop over all models, create features from forward passes and reduce dims
    #trafo_type = 'umap' 'autoencoder' 'pca' 'mds'
    all_trafo_types = ['pca'] #['umap', 'autoencoder', 'mds']
    filter_names = [
                    'mean',
                    # '1d-pca',
                    # 'bold-kernel-1',
                    # 'bold-kernel-2',
                    # 'bold-kernel-3'
                    ]
    num_components = [50, 100]
    for filter_name in filter_names:
        for trafo_type in all_trafo_types:
            for model_type in all_models:
                save_dir = f'../data/features/{model_type}/{filter_name}'
                for num_comps in num_components:
                    run_compression(save_dir, model_type, trafo_type, num_comps)
