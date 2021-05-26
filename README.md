## First Steps to Get Started
1. Generate Alexnet Features and get feeling (code base/runtime)
2. How much variance explained by PCA - Use larger dim.
3. Run linear regression encoding.
4. Make first submission with Alexnet
5. Look at fMRI data - variation across subjects (same video/ROI)
6. Add regularization to linear regression fit

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
8. Cross-Validation Setup (5 fold - 80-20 splits)

## Questions:
- What are repetitions? And what does this line of code do?
`ROI_data_train = np.mean(ROI_data["train"], axis = 1)`
- Is this the time averaged signal? Can we use this as auxiliary tasks?
- How many voxels are there per ROI? Different across subjects?
- At what level can model share data? All subject core or all ROI core?

## Data Exploration
- Heterogeneity across subjects and ROIs
    - Responses to same video
    - Voxels recorded

## Experiments to run
- What is the right sampling rate?
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

## Step 1: Extract Alexnet features for the videos

```
python feature_extraction/generate_features_alexnet.py
```

* This code saves Alexnet features for every frame of every video, as well as a PCA transformation of these features to get top-100 components. These activations are split in train and test data

## Step 2: Predict fMRI responses

```
python perform_encoding.py
```

* With the default arguments, the script expects a directory ````./participants_data_v2021```` with real fMRI data and ````./alexnet/```` with extracted NN features. It will generate predicted features using Alexnet (````--model````) layer_5 (````--layer````) for EBA (````--roi````) of subject 4 (````--sub````) in validation mode (````--mode````). The results will be stored in a directory called ````./results````. Running the script in default mode should return a mean correlation of 0.23

## Step 3: Prepare Submission

```
python prepare_submission.py
```

* With the default arguments, the script expects the results from step 2 in a directory ```./results/alexnet_devkit/layer_5```. It prepares the submission for all 9 ROIs (```mini_track```) . To generate results for ```full_track``` change the arguments as mentioned above.
* The script creates a Pickle and a zip file (containing the Pickle file) for the corresponding track that can then be submitted for participation in the challenge.
* Submit the ```mini_track``` results <a href="https://competitions.codalab.org/competitions/30930?secret_key=0d92787c-69d7-4e38-9780-94dd3a301f6b#participate-submit_results">here</a> and ```full_track``` results <a href="https://competitions.codalab.org/competitions/30937?secret_key=f3d0f352-c582-49cb-ad7c-8e6ec9702054#participate-submit_results">here</a>
