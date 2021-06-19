mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 AlexNet ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_alexnet.json \
    --experiment_dir experiments/roi/pls_pca_50_alexnet_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 VGG-19 ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_vgg.json \
    --experiment_dir experiments/roi/pls_pca_50_vgg_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNet18 ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_resnet18.json \
    --experiment_dir experiments/roi/pls_pca_50_resnet18_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNet34 ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_resnet34.json \
    --experiment_dir experiments/roi/pls_pca_50_resnet34_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNet101 ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_resnet101.json \
    --experiment_dir experiments/roi/pls_pca_50_resnet101_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNet152 ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_resnet152.json \
    --experiment_dir experiments/roi/pls_pca_50_resnet152_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 Efficientnet-B3 ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_efficientnet_b3.json \
    --experiment_dir experiments/roi/pls_pca_50_efficientnet_b3_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNext ROI BO 25 \
    --base_train_config configs/train/roi/pls_pca_50_resnext50_32x4d.json \
    --experiment_dir experiments/roi/pls_pca_50_resnext50_32x4d_bo_25/
