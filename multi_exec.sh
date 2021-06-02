mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet AlexNet BO 25 \
    --base_train_config configs/train/elastic_alexnet.json \
    --experiment_dir experiments/elastic_alexnet_bo_25/
mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet ResNet50 BO 25 \
    --base_train_config configs/train/elastic_resnet50.json \
    --experiment_dir experiments/elastic_resnet50_bo_25/
mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet EfficientNet BO 25 \
    --base_train_config configs/train/elastic_efficientnet_b3.json \
    --experiment_dir experiments/elastic_efficientnet_b3_bo_25/
mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet resnext50_32x4d BO 25 \
    --base_train_config configs/train/elastic_resnext50_32x4d.json \
    --experiment_dir experiments/elastic_resnext50_32x4d_bo_25/
