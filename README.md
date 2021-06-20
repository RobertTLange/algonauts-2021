# Algonauts-2021 Challenge Solution

![](docs/pipeline.png)

## Installation & Reproducing The Results

Install all required dependencies (in a clean environment):
```
pip install -r requirements.txt
```

Generate activations from different neural network architectures (AlexNet, VGG, ResNets, VOneNetworks and SimCLR-v2):
```
cd feature_extraction
python generate_features.py
cd ../feature_compression
python run_compression.py
```

Run on Bayesian Optimization tuning (50 iterations) for Subject 1 and ROI V1 with AlexNet PCA-100 features and ElasticNet encoding:
```
python run_roi.py -config_fname configs/train/base_config.json
```

Running every single configuration sequentially can take a long time. We parallelize the general Encoding-10-Fold-CV-BayesOpt pipeline over subjects and ROIs. In order to do so efficiently we rely on the [`MLE-Toolbox`](https://github.com/RobertTLange/mle-toolbox). It provides a framework for logging and scheduling experiments on either Slurm, OpenGridEngine or Google Cloud Platform. A grid search over subjects and ROIs can then be launched via:

```
mle run configs/cluster/base_roi.yaml
```

### Visualizing The Results

```
jupyter lab notebooks/inspect_experiments.ipynb
```

### Creating A Submission

```
jupyter lab notebooks/manual_submit.ipynb
```
