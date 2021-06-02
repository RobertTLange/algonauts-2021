mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose ElasticNet VGG ROI BO 50 \
    --base_train_config configs/train/roi/elastic_vgg.json \
    --experiment_dir experiments/roi/elastic_vgg_bo_50/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose ElasticNet AlexNet ROI BO 50 \
    --base_train_config configs/train/roi/elastic_alexnet.json \
    --experiment_dir experiments/roi/elastic_alexnet_bo_50/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose ElasticNet ResNet50 ROI BO 50 \
    --base_train_config configs/train/roi/elastic_resnet50.json \
    --experiment_dir experiments/roi/elastic_resnet50_bo_50/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose ElasticNet EfficientNet ROI BO 50 \
    --base_train_config configs/train/roi/elastic_efficientnet_b3.json \
    --experiment_dir experiments/roi/elastic_efficientnet_b3_bo_50/
mle run configs/cluster/base_roi.yaml \
    --no_welcome \
    --purpose ElasticNet resnext50_32x4d ROI BO 50 \
    --base_train_config configs/train/roi/elastic_resnext50_32x4d.json \
    --experiment_dir experiments/roi/elastic_resnext50_32x4d_bo_50/
