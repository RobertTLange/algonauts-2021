mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS 100 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_100_resnet50.json \
    --experiment_dir experiments/roi/pls_100_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS 500 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_500_resnet50.json \
    --experiment_dir experiments/roi/pls_500_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS Raw ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_raw_resnet50.json \
    --experiment_dir experiments/roi/pls_raw_resnet50_bo_25/
