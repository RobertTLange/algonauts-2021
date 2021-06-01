mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet VGG-19 BO 50 \
    --base_train_config configs/train/elastic_vgg.json \
    --experiment_dir experiments/elastic_vgg_bo_50/
mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet AlexNet BO 50 \
    --base_train_config configs/train/elastic_alexnet.json \
    --experiment_dir experiments/elastic_alexnet_bo_50/
mle run configs/cluster/base_bo.yaml \
    --no_welcome \
    --purpose ElasticNet ResNet50 BO 50 \
    --base_train_config configs/train/elastic_resnet50.json \
    --experiment_dir experiments/elastic_resnet50_bo_50/
