# mle run configs/cluster/autosklearn_roi.yaml \
#     --no_welcome \
#     --purpose Layerwise Autosklearn ROI \
#     --base_train_config configs/train/roi/auto_resnet50.json \
#     --experiment_dir experiments/roi/layerwise_auto/
# mle run configs/cluster/autosklearn_wb.yaml \
#     --no_welcome \
#     --purpose  Layerwise Autosklearn WB \
#     --base_train_config configs/train/wb/auto_resnet50.json \
#     --experiment_dir experiments/wb/layerwise_auto/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose PLS PCA 75 SimCLR - Mean - 25 Layers
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 60 SimCLR - Mean - 25 Layers \
    --base_train_config configs/train/roi/more_layers/pls_pca_60_simclr_r50_mean.json \
    --experiment_dir experiments/roi/more_layers/pls_pca_60_simclr_r50_mean/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 70 SimCLR - Mean - 25 Layers \
    --base_train_config configs/train/roi/more_layers/pls_pca_70_simclr_r50_mean.json \
    --experiment_dir experiments/roi/more_layers/pls_pca_70_simclr_r50_mean/
