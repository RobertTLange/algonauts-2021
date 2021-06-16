import os
import glob
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA


def do_dim_reduction_and_save(activations_dir, save_dir, num_pca_dims):
    onlyfiles = [f for f in os.listdir(activations_dir)
                 if os.path.isfile(os.path.join(activations_dir, f))]
    num_layers = int(len(onlyfiles)/1102)
    print("Number of layers", num_layers)
    layers = ['layer_' + str(l+1) for l in range(num_layers)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        explained_variance = []
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
        print(x_train.shape)
        # TODO: General fitting of dimensionality reduction technique
        # Full vs incremental (depending on pca dim and sampling rate)
        pca = PCA(n_components=num_pca_dims)#, batch_size=20)
        pca.fit(x_train)
        #explained_variance.append(pca.explained_variance_ratio_.cumsum()[-1])

        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        #print(x.shape, x_train.shape, x_test.shape)
        train_save_path = os.path.join(save_dir, "train_" + layer)
        test_save_path = os.path.join(save_dir, "test_" + layer)
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)
    pca_var_path = os.path.join(save_dir, "pca_variance")
    np.save(pca_var_path, np.array(explained_variance))


def run_compression(save_dir):
    pca_dims = [100, 250, 500]
    activations_dir = os.path.join(save_dir, "activations")
    # preprocessing using PCA and save
    for num_pca_dims in pca_dims:
        pca_dir = os.path.join(save_dir, f'pca_{num_pca_dims}')
        print(f"------performing  PCA: {num_pca_dims}---------")
        do_dim_reduction_and_save(activations_dir, pca_dir, num_pca_dims)


if __name__ == "__main__":
    all_models = [
                  'alexnet',
                  #'vgg',
                  #'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
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
    pca_dims = [100, 250, 500]
    for model_type in all_models:
        save_dir = f'../data/features/{model_type}'
        run_compression(save_dir)
