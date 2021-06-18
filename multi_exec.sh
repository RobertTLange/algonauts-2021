mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_50_resnet50.json \
    --experiment_dir experiments/roi/pls_50_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS UMAP 100 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_umap_100_resnet50.json \
    --experiment_dir experiments/roi/pls_umap_100_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS UMAP 250 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_umap_250_resnet50.json \
    --experiment_dir experiments/roi/pls_umap_250_resnet50_bo_25/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS UMAP 500 ResNet50 ROI BO 25 \
    --base_train_config configs/train/roi/pls_umap_500_resnet50.json \
    --experiment_dir experiments/roi/pls_umap_500_resnet50_bo_25/
