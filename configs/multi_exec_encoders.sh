mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose OLS PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/ols_pca_50_resnet50.json \
    --experiment_dir experiments/roi/ols_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose Elastic PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/elastic_pca_50_resnet50.json \
    --experiment_dir experiments/roi/elastic_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose MLP PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/mlp_pca_50_resnet50.json \
    --experiment_dir experiments/roi/mlp_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose CCA PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/cca_pca_50_resnet50.json \
    --experiment_dir experiments/roi/cca_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose C-PLS PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/cpls_pca_50_resnet50.json \
    --experiment_dir experiments/roi/cpls_pca_50_resnet50_bo_25/
