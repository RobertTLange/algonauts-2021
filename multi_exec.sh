mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose MLP ResNet50 ROI BO 50 \
    --base_train_config configs/train/roi/mlp_resnet50.json \
    --experiment_dir experiments/roi/mlp_resnet50_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose MLP ResNet50 WB BO 30 \
    --base_train_config configs/train/wb/mlp_resnet50.json \
    --experiment_dir experiments/wb/mlp_resnet50_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose MLP EfficientNet ROI BO 50 \
    --base_train_config configs/train/roi/mlp_efficientnet_b3.json \
    --experiment_dir experiments/roi/mlp_efficientnet_b3_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose MLP EfficientNet WB BO 30 \
    --base_train_config configs/train/wb/mlp_efficientnet_b3.json \
    --experiment_dir experiments/wb/mlp_efficientnet_b3_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose MLP resnext50_32x4d ROI BO 50 \
    --base_train_config configs/train/roi/mlp_resnext50_32x4d.json \
    --experiment_dir experiments/roi/mlp_resnext50_32x4d_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose MLP resnext50_32x4d WB BO 30 \
    --base_train_config configs/train/wb/mlp_resnext50_32x4d.json \
    --experiment_dir experiments/wb/mlp_resnext50_32x4d_bo_30/
