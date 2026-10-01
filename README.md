# Sargassum_segmentation
Training code for Sargassum segmentation and other classes in coastal images.

For compatibility with the required libraries, I recommend using a Python environment. The Yaml file could help to import the tested Python environment in Anaconda.

For training:

Select the paths of the training and validation datasets.
Select the model and backbone to use based on the PyTorch segmentation model framework.
Set the batch size and, if necessary, the accumulation steps when working with large models such as SegFormer MIT_B5 or DPT Tu_ViT_Large.
The training code was tested on almost all models, but DPT requires some changes to avoid NaN values in MIoU and consequently in loss values, so a version ready for training is provided.

For test and metrics:
Select the routes of the dataset and model, and modify the model and backbone. 
To ensure the correct file names, modify the MODEL_TYPE value for the current testing.
