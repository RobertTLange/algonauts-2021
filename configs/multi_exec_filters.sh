mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS 1D-PCA PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet50_1d_pca.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet50_1d_pca_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS BOLD 1 PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet50_bold_1.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet50_bold_1_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS BOLD 2 PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet50_bold_2.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet50_bold_2_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS BOLD 3 PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet50_bold_3.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet50_bold_3_bo_25/
#################################################################
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS 1D-PCA PCA 50 ResNet152 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet152_1d_pca.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet152_1d_pca_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS BOLD 1 PCA 50 ResNet152 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet152_bold_1.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet152_bold_1_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS BOLD 2 PCA 50 ResNet152 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet152_bold_2.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet152_bold_2_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS BOLD 3 PCA 50 ResNet152 ROI BO 25 \
    --base_train_config configs/train/roi/filters/pls_pca_50_resnet152_bold_3.json \
    --experiment_dir experiments/roi/filters/pls_pca_50_resnet152_bold_3_bo_25/
