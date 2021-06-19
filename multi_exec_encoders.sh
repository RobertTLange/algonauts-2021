mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 25 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_25_resnet50.json \
    --experiment_dir experiments/roi/pls_25_resnet50_bo_25/
# - OLS
# - Partial LS
# - Elastic Net
# - Residual MLP
# - CCA
# - PLS-SVD
