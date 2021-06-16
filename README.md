# Experiments to Run

- [ ] ResNet50 PCA-250 Partial LS 20 BO Iters

- [ ] Fix Partial LS + ResNet50
    - Different dim reduction techniques
        - PCA-100
        - PCA-250
        - PCA-500
        - UMAP-100
        - UMAP-250
        - UMAP-500
        - VAE-100
        - VAE-250
        - VAE-500

- [ ] Fix <Dim Reduction> + Partial LS
    - Different feature gen architectures
        - AlexNet
        - VGG-19
        - Resnet18
        - Resnet34
        - Resnet50
        - Resnet101
        - Resnet152
        - Efficientnet-b3

- [ ] Fix <Dim Reduction> + <Feature Architecture>
    - Different encoding models
        - OLS
        - Partial LS
        - Elastic Net
        - Residual MLP
        - GradientTreeRegressor

- [ ] VOne Networks
    - VOne-AlexNet
    - VOne-ResNet
    - VOne-ResNet-AT
    - VOne-ResNet-RS
    - VOne-CornetS

- [ ] 9 SimCLR-v2 Networks

- [ ] ImageNet Validation Scores

## Important Notes:
- Need to upload result .pkl as .zip without subdirectories being zipped!
- Can get detailed results for all regions via `Download output from scoring step`

## Questions:
- Different dim. reduction techniques (PCA dims/PWCCA)?
- What are repetitions? And what does this line of code do?
`ROI_data_train = np.mean(ROI_data["train"], axis = 1)`
- In `generate_features.py` all features are being averaged. Is this the correct thing to do? Or is there a smarter more fine-grained version
- Is this the time averaged signal? Can we use this as auxiliary tasks?
- How many voxels are there per ROI? Different across subjects?
- At what level can model share data? All subject core or all ROI core?
- What is the right sampling rate?
- Single layer vs full feature selection from all?
- What model fits best for which ROI and subject?
- How to combine features of different layers/networks?
- Finetune feature extractor on neural data

## Data Exploration
- Heterogeneity across subjects and ROIs
    - Responses to same video
    - Voxels recorded
- What categories are in the train/test set?
    - ImageNet predictions/Manual
    - Fine-tuning on these?

## Modelling - Mainly on Encoding Side!
- Shared Resnet core + Subject-specific + ROI-specific output heads
- FFW vs. LSTM and which level recurrence?
- XGBoostRegressor subject/roi level

## Links:
- XGBoost Regression: https://github.com/dmlc/xgboost/tree/master/demo/CLI/regression
- XGBoost CV: https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
- Sklearn Multioutput: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
- Multioutput XGBoost: https://gist.github.com/MLWave/4a3f8b0fee43d45646cf118bda4d202a
- BayesOpt XGBoost: https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
- BayesOpt API: https://github.com/fmfn/BayesianOptimization
- Pretrained ViT: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
- Multioutput cross validation: https://machinelearningmastery.com/multi-output-regression-models-with-python/
- Teapot evo algo: http://epistasislab.github.io/tpot/
- Torchvision models: https://pytorch.org/vision/stable/models.html
- Alexnet torch vision: https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html#alexnet
- Orthogonal Matching Pursuit: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
