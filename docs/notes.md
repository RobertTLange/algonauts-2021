## Need to Run Before Group Meeting

- [x] Get activations
    - All architectures: Mean
    - ResNet50: HRF Filters 1,2,3

- [x] Get dim reduction
    - All architectures: PCA
    - ResNet50: UMAP, MDS, VAE

- [x] Run compression: ResNet50 + PLS
- [x] Run architectures: PCA-50
- [x] Run encoders: Architecture-ResNet152
- [ ] Run temporal filters (mean, 1d-pca, hrf-1, hrf-2, hrf-3)
- [ ] Run VOneNetworks
- [ ] Run SimCLR-v2

## Ideas to Try Out

- Use more residual layers/finer resolution - relu/pooling
- Cross-validate range of HRF filters
- Rerun with ResNet152 encoding + orthogonal matching
- Try doing decoding from imagenet video labels
    - use to inform encoding models?!
- Try combining predictions across datasets/encoding models

## Notes and Questions:

- Need to upload result .pkl as .zip without subdirectories being zipped!
- Can get detailed results for all regions via `Download output from scoring step`
- Only using 5 ResNet layers, but there are many more. What resolution to pick?
- What sampling rate and temporal feature aggregation makes sense (weighted mean)?
- LOC encodes movement: Does it make sense to model differences in features?
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
- What categories are in the train/test set?
    - ImageNet predictions/Manual
    - Fine-tuning on these?
- Shared Resnet core + Subject-specific + ROI-specific output heads
- FFW vs. LSTM and which level recurrence?

- Convolving with HRF: https://bic-berkeley.github.io/psych-214-fall-2016/convolution_background.html
- Pretrained ViT: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py - how to get features
- Orthogonal Matching Pursuit: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
