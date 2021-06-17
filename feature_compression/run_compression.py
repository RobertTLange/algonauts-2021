import os
import glob
from tqdm import tqdm
import numpy as np
from fit_pca import fit_trafo_pca
from fit_umap import fit_trafo_umap
from fit_autoencoder import fit_trafo_autoencoder
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
            x[i,:] = temp
        x_train = x[:1000,:]
        x_test = x[1000:]
        #print(x.shape, x_train.shape, x_test.shape)S
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)

        # TODO: General fitting of dimensionality reduction technique
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
        #print(x_train.shape, x_train_trafo.shape)
        train_save_path = os.path.join(save_dir, "train_" + layer)
        test_save_path = os.path.join(save_dir, "test_" + layer)
        np.save(train_save_path, x_train_trafo)
        np.save(test_save_path, x_test_trafo)

        all_layers_info[layer] = info_l
    info_path = os.path.join(save_dir, info_title)
    save_pkl_object(all_layers_info, info_path)
    print(all_layers_info)


def run_compression(save_dir, trafo_type, info_title):
    pca_dims = [100, 250, 500, 750]
    umap_params ={"n_neighbors": 15,
                  "min_dist": 0.1,
                  "n_components": 2,
                  "metric":'euclidean'}
    activations_dir = os.path.join(save_dir, "activations")
    # preprocessing using PCA and save
    for num_pca_dims in pca_dims:
        info_title += f'_{num_pca_dims}.pkl'
        dim_red_dir = os.path.join(save_dir, f'{trafo_type}_{num_pca_dims}')
        print(f"------performing  {trafo_type}: {num_pca_dims}---------")
        dim_red_params = {"n_components": num_pca_dims}
        do_dim_reduction_and_save(activations_dir,
                                  dim_red_dir,
                                  trafo_type,
                                  dim_red_params,
                                  info_title)


if __name__ == "__main__":
    all_models = [
                  #'alexnet',
                  #'vgg',
                  'resnet50'
                  #'resnet18', 'resnet34', 'resnet101', 'resnet152',
                  #'efficientnet_b3', 'resnext50_32x4d',
                  #"vone-alexnet",
                  #"vone-resnet50",
                  #"vone-resnet50_at",
                  #"vone-resnet50_ns",
                  #"vone-cornets",
                  # 'simclr_r50_1x_sk0_100pct', 'simclr_r50_1x_sk0_10pct', 'simclr_r50_1x_sk0_1pct',
                  # 'simclr_r50_2x_sk1_100pct', 'simclr_r50_2x_sk1_10pct', 'simclr_r50_2x_sk1_1pct',
                  # 'simclr_r152_3x_sk1_100pct', 'simclr_r152_3x_sk1_10pct', 'simclr_r152_3x_sk1_1pct'
                  ]

    # Loop over all models, create features from forward passes and reduce dims
    trafo_type = 'pca'
    for model_type in all_models:
        save_dir = f'../data/features/{model_type}'
        run_compression(save_dir,
                        trafo_type=trafo_type,
                        info_title=f'{model_type}_{trafo_type}')
