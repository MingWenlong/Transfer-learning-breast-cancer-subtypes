# README

Predicting hormone receptors and PAM50 subtypes of breast cancer from multi-scale lesions of DCE-MR images using transfer learning models

# Schema

![1661852510240](https://user-images.githubusercontent.com/96223873/187404769-a6553c85-e71d-49e4-91ae-8694da8bda23.png)


# Installation requirements

* Software

		Python 3.5 +
				
* Packages (python)

		SimpleITK 1.2.0
		tensorflow 2.3.0
		keras 2.4.3
		connected-components-3d 3.1.2 
		scikit-image 0.15.0
		scikit-learn 0.22.2
		math 0.0.1
		pandas 0.24.2
		numpy 1.16.3


# Files description and Usage


* MRI_data_processing.py 

  		This python file details the MRI data preprocessing methods and steps in this work (from 3D NRRD file to 2D RBG images), including 5 steps:

  		(1) calculate the largest 3D connected component based on 26-adjacent and use as the volume-of-interest (VOI);
  		(2) tumor masks and peri-tumor (1.5mm, 3mm, 5mm, and 10mm) masks expansion based on the 3Dcc masks;
  		(3) N4 bias correction for 3T MRIs data;
  		(4) normalize the signal intensities of images to the range of 0 to 255;
  		(5) resize slice images to 224×224×3 RGB image for subsequent TF models.

* Transfer-learning-BC-clinical-receptors-prediction-5CV-tuning.py

		This python file details the transfer learning-based models for HR (ER and PR) classification prediction.
		
		In our scenario, 5-fold cross-validation (CV) was performed to avoid overfitting. 
		Data augmentation was applied for each slice image in training set by random rotation, translation, zoom, and flip. 
		Then 5 representative network architectures pre-trained on ImageNet were chosen (Inception-v3, Xception, Inception-ResNet-v2, VGG16 and ResNet50).
		
		ACC, AUROC, and AUPRC were employed to assess the prediction performance of testing dataset in the 5-fold CV.
		
* Transfer-learning-BC-PAM50-subtypes-prediction-5CV-tuning.py

		This python file details the transfer learning-based models for PAM50 multi-classification prediction.
		
		In our scenario, 5-fold cross-validation (CV) was performed to avoid overfitting. 
		Data augmentation was applied for each slice image in training set by random rotation, translation, zoom, and flip. 
		Then 5 representative network architectures pre-trained on ImageNet were chosen (Inception-v3, Xception, Inception-ResNet-v2, VGG16 and ResNet50).
		
		ACC, AUROC, AUPRC, and Top-2 ACC were employed to assess the prediction performance of testing dataset in the 5-fold CV.

* Example_data

  		The folder structures for TF model inputs and example images can be found in this folder.


# Citation

To be updated.

# License

Copyright (C) 2022 RadAI-Ming. Licensed GPLv3 for open source use.



