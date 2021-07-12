mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 SimCLR - Mean - 25 Layers \
    --base_train_config configs/train/roi/more_layers/pls_pca_50_simclr_r50_mean.json \
    --experiment_dir experiments/roi/more_layers/pls_pca_50_simclr_r50_mean/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose PLS PCA 50 SimCLR - Median - 25 Layers \
    --base_train_config configs/train/roi/more_layers/pls_pca_50_simclr_r50_median.json \
    --experiment_dir experiments/roi/more_layers/pls_pca_50_simclr_r50_median/
