## Ideas to Try Out

- Use different features for temporal dynamics
    - PCA time + PCA videos
    - Tensor Composition Analysis
    - Multilinear PCA
    - BOLD Filter Convolution of Features
- Use more residual layers
- Rerun with ResNet152 encoding + orthogonal matching
- Try doing decoding from imagenet video labels

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
