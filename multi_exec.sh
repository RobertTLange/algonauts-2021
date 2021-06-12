mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic Vone-ResNet50 ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vone_resnet50.json \
    --experiment_dir experiments/roi/elastic_vone_resnet50_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic Vone-ResNet50 WB BO 30 \
    --base_train_config configs/train/wb/elastic_vone_resnet50.json \
    --experiment_dir experiments/wb/elastic_vone_resnet50_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic Vone-ResNet50-AT ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vone_resnet50_at.json \
    --experiment_dir experiments/roi/elastic_vone_resnet50_at_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic Vone-ResNet50-AT WB BO 30 \
    --base_train_config configs/train/wb/elastic_vone_resnet50_at.json \
    --experiment_dir experiments/wb/elastic_vone_resnet50_at_bo_30/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose elastic Vone-Cornets ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vone_cornets.json \
    --experiment_dir experiments/roi/elastic_vone_cornets_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose elastic Vone-Cornets WB BO 30 \
    --base_train_config configs/train/wb/elastic_vone_cornets_at.json \
    --experiment_dir experiments/wb/elastic_vone_cornets_bo_30/
