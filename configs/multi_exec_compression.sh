mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_pca_50_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 100 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_pca_100_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_pca_100_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 250 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_pca_250_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_pca_250_resnet50_bo_25/
####################################################################################
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS UMAP 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_umap_50_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_umap_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS UMAP 100 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_umap_100_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_umap_100_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS UMAP 250 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_umap_250_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_umap_250_resnet50_bo_25/
####################################################################################
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS MDS 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_mds_50_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_mds_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS MDS 100 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_mds_100_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_mds_100_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS MDS 250 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_mds_250_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_mds_250_resnet50_bo_25/
####################################################################################
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS Autoencoder 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_autoencoder_50_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_autoencoder_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS Autoencoder 100 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_autoencoder_100_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_autoencoder_100_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS Autoencoder 250 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/compression/pls_autoencoder_250_resnet50.json \
    --experiment_dir experiments/roi/compression/pls_autoencoder_250_resnet50_bo_25/
