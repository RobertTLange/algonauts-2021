mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose ElasticNet ResNet50 WB BO 30 \
    --base_train_config configs/train/wb/elastic_resnet50.json \
    --experiment_dir experiments/wb/elastic_resnet50_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose ElasticNet AlexNet WB BO 30 \
    --base_train_config configs/train/wb/elastic_alexnet.json \
    --experiment_dir experiments/wb/elastic_alexnet_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose ElasticNet EfficientNet WB BO 30 \
    --base_train_config configs/train/wb/elastic_efficientnet_b3.json \
    --experiment_dir experiments/wb/elastic_efficientnet_b3_bo_50/
mle run configs/cluster/base_wb.yaml \
    --no_welcome \
    --purpose ElasticNet resnext50_32x4d WB BO 30 \
    --base_train_config configs/train/wb/elastic_resnext50_32x4d.json \
    --experiment_dir experiments/wb/elastic_resnext50_32x4d_bo_50/
