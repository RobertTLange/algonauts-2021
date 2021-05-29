## Infrastructure
1. Generate VGG and different ResNet Features once (high sampling?)
2. Different dim. reduction techniques (PCA dims/PWCCA)
3. Data Loader (Time Series and Simple FFW - grid over!)
    - Should be subject and roi specific!
    - Should be layer-specific as well?
4. Time/positional encoding feature for different frames
5. Train/Val setup (90-10 split)
6. Retrain encoding networks on full data for submission
7. Submission ready - multiprocessing pipeline (9 + 1 ROIs x 10 subjects)
8. Cross-Validation Setup (10 fold - 90-10 splits) - JAX decorator?
9. Automated pipeline that generates submission every morning for both tracks?
    - Need some form of hyper-hyper log with all results!
10. Different hyperopt pipelines: Grid, BO, Teapot

## Questions:
- What are repetitions? And what does this line of code do?
`ROI_data_train = np.mean(ROI_data["train"], axis = 1)`
- In `generate_features.py` all features are being averaged. Is this the correct thing to do? Or is there a smarter more fine-grained version
- Is this the time averaged signal? Can we use this as auxiliary tasks?
- How many voxels are there per ROI? Different across subjects?
- At what level can model share data? All subject core or all ROI core?

## Data Exploration
- Heterogeneity across subjects and ROIs
    - Responses to same video
    - Voxels recorded
- What categories are in the train/test set?
    - ImageNet predictions/Manual
    - Fine-tuning on these?

## Experiments to run
- What is the right sampling rate?
    - Can we match with timepoints of fMRI
- What layer fits best for which ROI and subject?
    - Single layer vs full feature selection from all
- What model fits best for which ROI and subject?
- How to combine features of different layers/networks?
- Run grid searches and Bayesian optimization on CPU/GPU?

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
