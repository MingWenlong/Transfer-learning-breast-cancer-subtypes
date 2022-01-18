# Transfer-learning-BC-PAM50-subtypes-prediction-5CV-tuning

import os
import shutil
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.utils import class_weight
import warnings

import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, models, metrics
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import time

print('start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

def transImgsBetFolders(src, dst, splitRate):
	"""
	transmission image files between folders
	"""
	global srcFiles
	srcFiles = os.listdir(src)
	if srcFiles:
		transFileNums = int(len(srcFiles) * splitRate)
		transIndex = random.sample(range(0, len(srcFiles)), transFileNums)
		for eachIndex in transIndex:
			shutil.move('{}{}'.format(src, str(srcFiles[eachIndex])), '{}{}'.format(dst, str(srcFiles[eachIndex])))
	else:
		print("No image file moved. SrcFiles is empty.")


def transImgsWithLabelBetFolders(src, dst, splitRate, labels):
	"""
	transmission image files with label info between folders
	"""
	for label in labels:
		transImgsBetFolders('{}{}/{}/'.format(datasetsFolderName, src, label), 
			'{}{}/{}/'.format(datasetsFolderName, dst, label) , splitRate)


def getImgsNameWithLabel_PAM(folderName, labels):
	"""
	get image file name with label info for PAM50 task
	"""
	srcFiles = os.listdir('{}train/{}'.format(datasetsFolderName, folderName))
	for val in srcFiles:
		X.append(val)
		if(folderName == labels[0]):
			Y.append(0)
		elif(folderName == labels[1]):
			Y.append(1)
		elif(folderName == labels[2]):
			Y.append(2)
		elif(folderName == labels[3]):
			Y.append(3)
		else:
			Y.append(4)


def get_Model(img_rows, img_cols, actFun, outActFun, kernel_ini, bias_ini, learnRate, neurons):
	# Instantiate a base model with pre-trained weights
	base_model = keras.applications.ResNet50(include_top = False, weights = "Path_To_Model_Weights", input_shape = (img_rows, img_cols, 3))
	base_model.trainable = False
	# Create a new model on top
	inputs = keras.Input(shape = (img_rows, img_cols, 3))
	x = base_model(inputs, training=False)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(neurons[0],activation = actFun,kernel_initializer = kernel_ini,bias_initializer = bias_ini)(x)
	x = layers.Dense(neurons[1],activation = actFun,kernel_initializer = kernel_ini,bias_initializer = bias_ini)(x)
	x = layers.Dense(neurons[2],activation = actFun,kernel_initializer = kernel_ini,bias_initializer = bias_ini)(x)
	x = layers.Dense(neurons[3],activation = actFun,kernel_initializer = kernel_ini,bias_initializer = bias_ini)(x)
	outputs = layers.Dense(5, activation = outActFun)(x)
	model = models.Model(inputs, outputs)
	model.compile(optimizer = optimizers.Adam(lr = learnRate),loss="categorical_crossentropy",
		metrics=['accuracy', metrics.AUC(curve="ROC", name="AUROC", multi_label = False), metrics.AUC(curve="PR", name="AUPRC", multi_label = False), 
		metrics.TopKCategoricalAccuracy(k=2)]
		)
	return model


def model_PAM50_skf_cv(X, Y, kFold = 5, batch_size = 32, epoch = 10, para_list = ['relu', 'softmax', 'glorot_uniform', 'zeros', 0.0001, [4096, 2048, 1024, 512]]):
	skf = StratifiedKFold(n_splits = kFold, shuffle = True)
	skf.get_n_splits(X, Y)
	foldNum = 0
	key_list = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'AUROC', 'val_AUROC', 
	'AUPRC', 'val_AUPRC', 'top_k_categorical_accuracy', 'val_top_k_categorical_accuracy']
	hist_results = dict([(k,[]) for k in key_list])
	# kflod CV
	for train_index, val_index in skf.split(X, Y):
			# first, move all MRI  from  validation folder to train folder before each fold train
			# including different labels
			transImgsWithLabelBetFolders('valid', 'train', 1, labels = img_labels_pam50)
			foldNum += 1
			print("<======== Results for fold: {} ========>".format(foldNum))
			print("batch_size: {}".format(batch_size))
			X_train, X_val = X[train_index], X[val_index]
			Y_train, Y_val = Y[train_index], Y[val_index]
			# then, move validation images of this folder from train folder to valid folder
			for eachIndex in range(len(X_val)):
				reLabel = ''
				if(Y_val[eachIndex] == 0):
					reLabel = img_labels_pam50[0]
				elif(Y_val[eachIndex] == 1):
					reLabel = img_labels_pam50[1]
				elif(Y_val[eachIndex] == 2):
					reLabel = img_labels_pam50[2]
				elif(Y_val[eachIndex] == 3):
					reLabel = img_labels_pam50[3]
				else:
					reLabel = img_labels_pam50[4]
				shutil.move('{}train/{}/{}'.format(datasetsFolderName, reLabel, X_val[eachIndex]), 
					'{}valid/{}/{}'.format(datasetsFolderName, reLabel, X_val[eachIndex]))
			# data agumentation for training data
			train_datagen = ImageDataGenerator(
				rescale = 1/255.0, 
				rotation_range = 20, 
				zoom_range = 0.05, 
				width_shift_range = 0.05, 
				height_shift_range = 0.05, 
				shear_range = 0.05, 
				horizontal_flip = True, 
				fill_mode = "nearest")
			valid_datagen = ImageDataGenerator(
				rescale = 1/255.0)
			#
			train_generator = train_datagen.flow_from_directory(
				directory = train_path,
				target_size = (img_rows, img_cols),
				batch_size = batch_size,
				color_mode = "rgb",
				class_mode = "categorical")
			valid_generator = valid_datagen.flow_from_directory(
				directory = valid_path,
				target_size = (img_rows, img_cols),
				batch_size = batch_size,
				color_mode = "rgb",
				class_mode = "categorical",
				shuffle = False)
			# unbanlanced
			class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
			class_weights = dict(enumerate(class_weights))
			# fit model
			model = get_Model(
				img_rows = img_rows, 
				img_cols = img_cols, 
				actFun = para_list[0], 
				outActFun = para_list[1], 
				kernel_ini = para_list[2], 
				bias_ini = para_list[3], 
				learnRate = para_list[4], 
				neurons = para_list[5])
			# result
			history = model.fit(
				train_generator,
				validation_data = valid_generator,
				class_weight = class_weights,
				epochs = epoch,
				verbose = 0)
			# output result for this flod
			for key in key_list:
				for hkey in list(history.history):
					if hkey == key:
						for item in history.history[hkey]:
							hist_results[key].append(item)
					elif hkey.startswith(key):
						for item in history.history[hkey]:
							hist_results[key].append(item)
					else:
						pass
	# results for all kflods
	return hist_results


warnings.filterwarnings("ignore")
# hyperParameters
# basic info for PAM50 task
neuron_big = [4096, 2048, 1024, 512]
neuron_small = [2048, 1024, 512, 128]
img_rows, img_cols =  224, 224
task = ['PAM50']
models_tf = ['InceptionV3', 'Xception', 'InceptionResNetV2', 'VGG16', 'ResNet50']
models_list = ['M0', 'M1', 'M2', 'M3', 'M4']
img_labels_pam50 = ['LumA', 'LumB','Her2E','Basal','Normal']
batch_size_list = [16, 32, 64, 128]
my_epoch = 100
evalu_idx_list = ['val_loss', 'val_accuracy', 'val_AUROC', 'val_AUPRC', 'val_top_k_categorical_accuracy']
# Requires specified information
model_select = models_tf[]
model_idx = models_list[]
datasetsFolderName = 'Path_To_datasets/Mask_{}/{}/'.format(model_idx, task) ##====================##
outputFilesPath = datasetsFolderName + 'Models/{}/'.format(model_select)


# 5CV training
srcFiles = []
# all train data
X = []
Y = []
getImgsNameWithLabel_PAM(img_labels_pam50[0], img_labels_pam50)
getImgsNameWithLabel_PAM(img_labels_pam50[1], img_labels_pam50)
getImgsNameWithLabel_PAM(img_labels_pam50[2], img_labels_pam50)
getImgsNameWithLabel_PAM(img_labels_pam50[3], img_labels_pam50)
getImgsNameWithLabel_PAM(img_labels_pam50[4], img_labels_pam50)
X = np.asarray(X)
Y = np.asarray(Y)
# dirs path
train_path = datasetsFolderName + 'train/'
valid_path = datasetsFolderName + 'valid/'
test_path = datasetsFolderName + 'test/'
# 5-flod CV used to get the best hyperParameters
# tuning batch_size and epoch
#
for batchSize in batch_size_list:
	BATEPO_res = model_PAM50_skf_cv(X, Y, kFold = 5, batch_size = batchSize, epoch = my_epoch)
	df_BATEPO = pd.DataFrame(BATEPO_res)
	df_BATEPO.index = ['batchSize{}_epoch{}_fold{}'.format(batchSize, epo + 1, fld + 1) for fld in range(5) for epo in range(my_epoch)]
	pd.DataFrame(df_BATEPO).to_csv('{}batchSize{}_epoch{}_PAM50_5CV_{}_trainingResults.csv'.format(outputFilesPath, batchSize, my_epoch, model_select))

