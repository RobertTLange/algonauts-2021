mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 25 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_25_resnet50.json \
    --experiment_dir experiments/roi/pls_25_resnet50_bo_25/
# - AlexNet
# - VGG-19
# - Resnet18
# - Resnet34
# - Resnet50
# - Resnet101
# - Resnet152
# - Efficientnet-b3
# - ResNext
