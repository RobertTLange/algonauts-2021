## First Steps to Get Started

1. Generate Alexnet Features and get feeling (code base/runtime)
2. How much variance explained by PCA - Use larger dim.
3. Run linear regression encoding.
4. Make first submission with Alexnet
5. Look at fMRI data - variation across subjects (same video/ROI)
6. Add regularization to linear regression fit

## Infrastructure

1. Generate VGG and different ResNet Features
2. Different dim. reduction techniques
3. Data Loader (Time Series and Simple FFW)
4. Time/positional encoding feature
5. Train/Val setup (cross-validation?)
6. Retrain full encoding networks for submission
7. Submission ready - multiprocessing pipeline (8 regions x 10 subjects)

## Experiments to run

- What is the right sampling rate?
- What layer fits best for which ROI and subject?
- What model fits best for which ROI and subject?
- How to combine features of different layers/networks?
- Run grid searches and Bayesian optimization on CPU/GPU?

## Modelling - Mainly on Encoding Side!

- Shared Resnet core + Subject-specific + ROI-specific output heads
- FFW vs. LSTM and which level recurrence?

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
