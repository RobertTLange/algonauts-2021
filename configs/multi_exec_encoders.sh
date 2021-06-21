mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose OLS PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/encoders/ols_pca_50_resnet50.json \
    --experiment_dir experiments/roi/encoders/ols_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose Elastic PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/encoders/elastic_pca_50_resnet50.json \
    --experiment_dir experiments/roi/encoders/elastic_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose MLP PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/encoders/mlp_pca_50_resnet50.json \
    --experiment_dir experiments/roi/encoders/mlp_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose CCA PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/encoders/cca_pca_50_resnet50.json \
    --experiment_dir experiments/roi/encoders/cca_pca_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose C-PLS PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/encoders/cpls_pca_50_resnet50.json \
    --experiment_dir experiments/roi/encoders/cpls_pca_50_resnet50_bo_25/
