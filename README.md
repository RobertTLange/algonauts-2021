# Algonauts-2021 Challenge (Mini-Track 5th Place)
## Author: [@RobertTLange](https://twitter.com/RobertTLange) [Last Update: August 2021]

This repository contains a solution approach to the [Algonauts Challenge 2021](http://algonauts.csail.mit.edu/), in which one had to predict fMRI-recorded voxel activity across different ROIs/subjects and for different small video snippet stimuli.

The final submitted solution is based on a SimCLR-v2 contrastive pre-trained ResNet-50, which was fine-tuned on supervised labels of ImageNet. The layer-wise features extracted from the videos were PCA-dimensionality reduced down to 50 features. Afterwards, we run a Bayesian Optimization (BO) Loop to tune the hyperparameters of a PLS regression encoding model. The BO procedure is layer-, subject- and ROI-specific and uses 10-fold cross-validation. Afterwards, we select the best performing encoding model and retrain on all video-voxel datapoints. The full pipeline is depicted below:

![](docs/pipeline.png)

## Installation & Reproducing The Results

Install all required dependencies (in a clean conda environment):
```
./setup.sh
```

Generate activations from different neural network architectures (AlexNet, VGG, ResNets, VOneNetworks or SimCLR-v2):
```
python run_features.py
```

Run on Bayesian Optimization tuning (50 iterations) for Subject 1 and ROI V1 with AlexNet PCA-100 features and ElasticNet encoding:
```
python run_bayes_opt.py -config_fname configs/train/base_config.json
```

Running every single configuration sequentially can take a long time. We parallelize the general `Encoding-10-Fold-CV-BayesOpt` pipeline over subjects and ROIs. In order to do so efficiently we rely on the [`MLE-Toolbox`](https://github.com/RobertTLange/mle-toolbox). It provides a framework for logging and scheduling experiments on either Slurm, OpenGridEngine or Google Cloud Platform. A grid search over subjects and ROIs can then be launched via:

```
mle run configs/cluster/base_roi.yaml
```

Finally, if you would like to run the Auto-Sklearn v.1.0 pipeline for a subject, you can do so via

```
python run_autosklearn.py -config_fname configs/train/roi/auto_resnet50.json
```

### Visualizing The Results

```
jupyter lab notebooks/inspect_experiments.ipynb
```

### Creating A Submission

```
jupyter lab notebooks/manual_submit.ipynb
```

### Credits & Acknowledgements

This repository makes use of several modified open source repositories. We include their respective license in the respective subdirectories. The code repositories include the following:

- [`SimCLR-v2-PyTorch` by Separius](https://github.com/Separius/SimCLRv2-Pytorch) - GPL-3.0 License.
- [`barlowtwins` by facebookresearch](https://github.com/facebookresearch/barlowtwins) - MIT License.
- [`vonenet` by dicarlolab](https://github.com/dicarlolab/vonenet) - GPL-3.0 License.

Finally, we are thankful to the [`timm` package](https://github.com/rwightman/pytorch-image-models) creators for providing easily accessible model checkpoints and the algonauts challenge organizers for putting together this challenge!
