mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic 250 Efficient ROI BO 50 \
    --base_train_config configs/train/roi/elastic_efficientnet_b3.json \
    --experiment_dir experiments/roi/elastic_250_efficientnet_b3_b50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic 250 Efficient WB BO 30 \
    --base_train_config configs/train/wb/elastic_efficientnet_b3.json \
    --experiment_dir experiments/wb/elastic_250_efficientnet_b3_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic 250 VGG ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vgg.json \
    --experiment_dir experiments/roi/elastic_250_vgg_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic 250 VGG WB BO 30 \
    --base_train_config configs/train/wb/elastic_vgg.json \
    --experiment_dir experiments/wb/elastic_250_vgg_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic 250 Vone-ResNet50-AT ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vone_resnet50_at.json \
    --experiment_dir experiments/roi/elastic_250_vone_resnet50_at_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic 250 Vone-ResNet50-AT WB BO 30 \
    --base_train_config configs/train/wb/elastic_vone_resnet50_at.json \
    --experiment_dir experiments/wb/elastic_250_vone_resnet50_at_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic 250 Vone-Cornets ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vone_cornets.json \
    --experiment_dir experiments/roi/elastic_250_vone_cornets_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic 250 Vone-Cornets WB BO 30 \
    --base_train_config configs/train/wb/elastic_vone_cornets.json \
    --experiment_dir experiments/wb/elastic_250_vone_cornets_bo_30/
