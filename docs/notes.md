- Higher sampling rates - How do these compare?!
    - Small effect on test performance - not on validation!
- Use median activation instead of mean
- Evaluate more layers of ResNet = all stem layers and all blocks
    - ResNet50 = 25 layers (vs 7 before)

## Ideas to Try Out

- Do contrastive learning directly on video frames?! Self-supervised finetuning
- Try doing decoding from imagenet video labels
    - use to inform encoding models?!
- Cross-validate range of HRF filters

- SSL - Barlow Twins architecture/BYOL
- Segmentation models YOLO or UNets

- Tuning Ideas
    - Use more residual layers/finer resolution - relu/pooling
    - Try ensembling predictions across datasets/encoding models
    - Pick best model per subject per roi = Create one larger dataframe 'hyperhyperlog'

- Things to check/engineer
    - Have a look at BO logs. What PLS hyperparams are found?

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
