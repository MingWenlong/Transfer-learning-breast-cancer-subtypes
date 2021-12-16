# /usr/bin/env python 3.5
# convert pre, post-1 and post-5 MR images to RGB three channel images for transfer learning

import SimpleITK as sitk
import os
import numpy as np
import cc3d
import copy
from skimage import io
import math


###
# P1, Get the 3D connected component for tumor masks
def loadfiles(filePath):
	"""
	load files
	"""
	mFiles = os.listdir(filePath)
	mFiles.sort()
	return(mFiles)


def getLargest3Dcc(labels_in, connectivity):
	"""
	get the largest 3d ROI label
	"""
	# default is the 26-connected for 3D np array
	labels_out, Ncc = cc3d.connected_components(labels_in, connectivity=connectivity, return_N=True)
	if Ncc == 0:
		print('wrong input!')
	elif Ncc == 1:
		return(labels_in) ## out is in
	else:
		segid_N = []
		for segid in range(1, Ncc+1):
			segid_N.append(np.sum(labels_out == segid))
		Lasegid = int(np.where(segid_N == np.max(segid_N))[0]) + 1
		extracted_image = labels_out * (labels_out == Lasegid)
		# filtered mask with the largest ROI
		extracted_image[extracted_image == Lasegid] = 1
		return(extracted_image)


def outnrrd(array, referImage):
	"""
	tranform array to sitk object
	"""
	image = sitk.GetImageFromArray(array)
	image.SetOrigin(referImage.GetOrigin())
	image.SetDirection(referImage.GetDirection())
	image.SetSpacing(referImage.GetSpacing())
	return(image)

# datasets analysis P1
maskPath = '/Path/To/Labels/'
outPath = '/Path/To/P1outputs/'
maskFile = loadfiles(filePath=maskPath)
for segid in maskFile:
	label = sitk.ReadImage(maskPath + segid)
	label_array = sitk.GetArrayFromImage(label).astype(np.uint8)
	# set the connectivity to 26
	New_label_array = getLargest3Dcc(labels_in = label_array, connectivity = 26)
	New_label = outnrrd(array = New_label_array, referImage = label)
	sitk.WriteImage(New_label, outPath + '3Dcc-' + segid, True)


###
# P2, expanding tumor masks based on the 3Dcc masks
def padsize(image, ROI):
	"""
	get the pad size for different ROI expand length (mm)
	"""
	ROI_X = image.GetSpacing()[0]
	ROI_Y = image.GetSpacing()[1]
	padsize_X = round(ROI/ROI_X)
	padsize_Y = round(ROI/ROI_Y)
	return(padsize_X, padsize_Y)


def getSlice(inputArray):
	"""
	get z axis
	"""
	ROI_Slice = []
	for i in range(0, inputArray.shape[0]): # [0] is z axis (usually is slices)
		if inputArray[i,:,:].sum() != 0:
			ROI_Slice.append(i)
	return(ROI_Slice)


def getYXindex(inputArray,inputSlice):
	"""
	get the position of ROIs (x,y) in slices (z),
	this function handles a series of slices
	"""
	YXindex = []
	for i in range(0, len(inputSlice)):
		voxel_index = np.where(inputArray[inputSlice[i],:,:] != 0)
		y_start = np.min(voxel_index[0]) # y axis start
		y_end = np.max(voxel_index[0]) + 1 # y axis end
		x_start = np.min(voxel_index[1]) # x axis start
		x_end = np.max(voxel_index[1]) + 1# x axis start
		YXindex.append([y_start, y_end, x_start, x_end])
	return(YXindex)


def padROIUnique(inputArray, interSizeX, interSizeY):
	"""
	pad ROI uniquely for each slice in each patients
	"""
	p_Slices = getSlice(inputArray = inputArray)
	p_YXs = getYXindex(inputArray = inputArray, inputSlice = p_Slices)
	pad_array = copy.deepcopy(inputArray)
	for i in range(0, len(p_Slices)):
		y_pad_start = p_YXs[i][0] - interSizeY
		y_pad_end = p_YXs[i][1] + interSizeY
		x_pad_start = p_YXs[i][2] - interSizeX
		x_pad_end = p_YXs[i][3] + interSizeX
		#
		pad_array[p_Slices[i], y_pad_start:y_pad_end, x_pad_start:x_pad_end] = 1
	return(pad_array)


# datasets analysis P2
inPath_mask = '/Path/To/3DccLabels/'
mask_files = loadfiles(filePath = inPath_mask)
# expand  x and y for each slice
outPath_ROI = '/Path/To/P2outputs/'
for i in range(0, len(mask_files)):
	mask = sitk.ReadImage(inPath_mask + mask_files[i])
	interSize = padsize(image = mask, ROI = 3)
	mask_array = sitk.GetArrayFromImage(mask) # format is z,y,x
	pad_ROI_array = padROIUnique(inputArray = mask_array, interSizeX = interSize[0], interSizeY = interSize[1])
	pad_mri = outnrrd(array = pad_ROI_array, referImage = mask)
	sitk.WriteImage(pad_mri, outPath_ROI + 'ROI_3mm-3Dcc-' + mask_files[i].split('-',2)[1] + '-mask.nrrd', True)


###
# P3, N4 bias correction for 3T MRIs based on different masks 
# datasets analysis P3
origin_MRIs_Path = '/Path/To/MRIs/'
mriFile = loadfiles(filePath = origin_MRIs_Path)
mris_pID_sub = []
for item in mriFile:
	if 'pre' in item or 'post_1'in item or 'post_3' in item or'post_5' in item:
		mris_pID_sub.append(item)

mris_pID_sub.sort()
# N4 bias correction
# based on 3D cc
Masks_Path = '/Path/To/3DccLabels/'
maskFile = loadfiles(filePath = Masks_Path)
outPath_N4 = '/Path/To/P3outputs/'
#
for i in range(0, len(maskFile)):
	maskImage = sitk.ReadImage(Masks_Path + maskFile[i])    # read mask image
	patientID = maskFile[i].split('-', 2)[1]    # patient ID
	# identify mri images for patientID
	patientImageList = []
	for j in range(0, len(mris_pID_sub)):
		if patientID in mris_pID_sub[j]:
			patientImageList.append(mris_pID_sub[j])
	shrinkFactor = int(1)
	for k in range(0, len(patientImageList)):
		inputImage = sitk.ReadImage(origin_MRIs_Path + patientImageList[k])  # read pre/post image
		inputImage = sitk.Shrink(inputImage, [shrinkFactor] * inputImage.GetDimension())
		maskImage = sitk.Shrink(maskImage, [shrinkFactor] * inputImage.GetDimension())
		maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
		inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
		# N4 bias correction
		corrector = sitk.N4BiasFieldCorrectionImageFilter()  # wiener filter for noise
		numberFittingLevels = 4
		outputN4 = corrector.Execute(inputImage, maskImage)
		sitk.WriteImage(outputN4, outPath_N4 + 'N4-3Dcc-' + patientImageList[k], True)


###
# P4, scale voxel value to [0, 255]
def IntensityScale(inputVoxels, outputMax, outputMin):
	"""
	scale ROI voxels' intnsity values
	"""
	outputVoxels = []
	for i in range(0, len(inputVoxels)):
		tmp = round((inputVoxels[i] - np.min(inputVoxels))*(outputMax - outputMin)/(np.max(inputVoxels) - np.min(inputVoxels)) + outputMin)
		outputVoxels.append(tmp)
	#
	return(outputVoxels)


def ValueNor_PrePostEPostL(preMRI, postEMRI, postLMRI, mask):
	"""
	scaled the pre, post early and post late MR images together, to contain the information between different MRIs
	"""
	# load images
	input_pre = sitk.ReadImage(preMRI)
	input_post_E = sitk.ReadImage(postEMRI)
	input_post_L = sitk.ReadImage(postLMRI)
	input_mask = sitk.ReadImage(mask)
	# get image array
	input_pre_array = sitk.GetArrayFromImage(input_pre)
	input_post_E_array = sitk.GetArrayFromImage(input_post_E)
	input_post_L_array = sitk.GetArrayFromImage(input_post_L)
	input_mask_array = sitk.GetArrayFromImage(input_mask)
	# merge image array
	input_merge_MRIs_array = np.concatenate((input_pre_array, input_post_E_array, input_post_L_array), axis = 0)
	input_merge_Masks_array = np.concatenate((input_mask_array, input_mask_array, input_mask_array), axis = 0)
	# get the ROI voxel values and index
	ROI_voxels = input_merge_MRIs_array[input_merge_Masks_array != 0]
	ROI_voxels_index = np.where(input_merge_Masks_array != 0)
	# scale the value
	ROI_voxels_scaled = IntensityScale(inputVoxels = ROI_voxels, outputMax = 255, outputMin = 0)
	# output the scaled voxels
	output_merge_MRIs_array = copy.deepcopy(input_merge_MRIs_array)
	# make non-ROI voxels' value to 0 
	output_merge_MRIs_array[input_merge_Masks_array == 0] = 0
	for i in range(0, len(ROI_voxels)):
		output_merge_MRIs_array[ROI_voxels_index[0][i], ROI_voxels_index[1][i], ROI_voxels_index[2][i]] = int(ROI_voxels_scaled[i])
	#
	return(output_merge_MRIs_array)


# datasets analysis P4
N4_3Dcc_MRIs_Path = '/Path/To/P3outputs/'
N4_3Dcc_Masks_Path = '/Path/To/3DccLabels/'
mriFile = loadfiles(filePath = N4_3Dcc_MRIs_Path)
maskFile = loadfiles(filePath = N4_3Dcc_Masks_Path)
outPath = '/Path/To/P4outputs/'
#
for i in range(0, len(maskFile)):
	patientID = maskFile[i].split('-', 2)[1]
	# identify mri images for patient' ID
	patientImageList = []
	for j in range(0, len(mriFile)):
		if patientID in mriFile[j]:
			patientImageList.append(mriFile[j])
	# image path
	preMRI_Path = N4_3Dcc_MRIs_Path + list(filter(lambda x: x.find("pre") >= 0, patientImageList))[0]
	postEMRI_Path = N4_3Dcc_MRIs_Path + list(filter(lambda x: x.find("post_1") >= 0, patientImageList))[0]
	postLMRI_Path = N4_3Dcc_MRIs_Path + list(filter(lambda x: x.find("post_5") >= 0, patientImageList))[0]
	#
	mask_Path = N4_3Dcc_Masks_Path + maskFile[i]
	# scaled to [0, 255] based on 'pre', 'post_1' and 'post_5'
	output_Merged_MRIs_Array = ValueNor_PrePostEPostL(preMRI = preMRI_Path, postEMRI = postEMRI_Path, postLMRI = postLMRI_Path, 
		mask = mask_Path)
	# split to different images
	output_Pre_Array, output_PostE_Array, output_PostL_Array = np.split(output_Merged_MRIs_Array, 3, axis = 0)
	# output images
	output_Pre_image = outnrrd(array = output_Pre_Array, referImage = sitk.ReadImage(preMRI_Path))
	output_PostE_image = outnrrd(array = output_PostE_Array, referImage = sitk.ReadImage(postEMRI_Path))
	output_PostL_image = outnrrd(array = output_PostL_Array, referImage = sitk.ReadImage(postLMRI_Path))
	#
	sitk.WriteImage(output_Pre_image, outPath + 'ValueScaled-N4-3Dcc-' + patientID + '-pre.nrrd', True)
	sitk.WriteImage(output_PostE_image, outPath + 'ValueScaled-N4-3Dcc-' + patientID + '-post_1.nrrd', True)
	sitk.WriteImage(output_PostL_image, outPath + 'ValueScaled-N4-3Dcc-' + patientID + '-post_5.nrrd', True)


###
# P5, convert to RGB images
def getpID(inputFile):
	"""
	get the patients ID from images
	"""
	pIDs = []
	for i in inputFile:
		pIDs.append(i.split('-')[3])
	pIDs = list(set(pIDs))
	pIDs.sort()
	return(pIDs)


def getImagebyPat(inputPatID, inputFile):
	patImage = []
	for i in inputFile:
		if inputPatID in i:
			patImage.append(i)
	return(patImage)


def getSlice(inputArray):
	"""
	get z axis
	"""
	ROI_Slice = []
	for i in range(0, inputArray.shape[0]): # [0] is z axis (usually is slices)
		if inputArray[i,:,:].sum() != 0:
			ROI_Slice.append(i)
	return(ROI_Slice)


def getYXindex(inputArray, inputSlice):
	"""
	get the position of ROIs (x,y) in slices (z),
	this function handles a series of slices
	"""
	YXindex = []
	for i in inputSlice:
		voxel_index = np.where(inputArray[i,:,:] != 0)
		y_start = np.min(voxel_index[0]) # y axis start
		y_end = np.max(voxel_index[0]) + 1 # y axis end
		x_start = np.min(voxel_index[1]) # x axis start
		x_end = np.max(voxel_index[1]) + 1# x axis start
		YXindex.append([y_start, y_end, x_start, x_end])
	return(YXindex)


def getLesionRegion(inputArray,inputSlice,inputYXindex):
	"""
	get the voxel value of ROI with the known slice and Y,X position
	"""
	Lesion3D = []
	if len(inputSlice) == len(inputYXindex):
		for i in range(0,len(inputSlice)):
			LesionArray = inputArray[inputSlice[i],inputYXindex[i][0]:inputYXindex[i][1],inputYXindex[i][2]:inputYXindex[i][3]]
			Lesion3D.append(LesionArray)
	return(Lesion3D)


def getPadarray(inputRGB3Darray,TFXsize,TFYsize):
	rgbArray_pad = []
	inputRGBsize = inputRGB3Darray.shape
	# y axis
	y_size = inputRGBsize[1]
	y_pad = int((TFYsize - y_size)/2)
	y_pad_left = math.floor((TFYsize - y_size)/2)
	y_pad_right = math.ceil((TFYsize - y_size)/2)
	# x axis
	x_size = inputRGBsize[2]
	x_pad = int((TFXsize - x_size)/2)
	x_pad_left = math.floor((TFXsize - x_size)/2)
	x_pad_right = math.ceil((TFXsize - x_size)/2)
	#
	if y_size%2 == 0 and x_size%2 == 0:
		rgbArray_pad = np.pad(inputRGB3Darray, ((0,0),(y_pad,y_pad),(x_pad,x_pad)), 'constant')
	elif y_size%2 != 0 and x_size%2 == 0:
		rgbArray_pad = np.pad(inputRGB3Darray, ((0,0),(y_pad_left,y_pad_right),(x_pad,x_pad)), 'constant')
	elif y_size%2 == 0 and x_size%2 != 0:
		rgbArray_pad = np.pad(inputRGB3Darray, ((0,0),(y_pad,y_pad),(x_pad_left,x_pad_right)), 'constant')
	else:
		rgbArray_pad = np.pad(inputRGB3Darray, ((0,0),(y_pad_left,y_pad_right),(x_pad_left,x_pad_right)), 'constant')
	return(rgbArray_pad)


def mergeYXindex(M1YXindex,M2YXindex,M3YXindex):
	Map_YXindex = []
	for i in range(0,len(M1YXindex)):
		M1_value,M2_value,M3_value = M1YXindex[i],M2YXindex[i],M3YXindex[i]
		M_value = []
		for j in range(0,4):
			tmp = np.max([M1_value[j],M2_value[j],M3_value[j]])
			M_value.append(tmp)
		Map_YXindex.append(M_value)
	return(Map_YXindex)


def getRGB(inputImage,contrastID,inputPath):
	patientRGB = []
	for i in inputImage:
		if contrastID[0] in i:
			Map1 = sitk.ReadImage(inputPath + i) # early PE map
			Map1_array = sitk.GetArrayFromImage(Map1) # format is z,y,x
		elif contrastID[1] in i:
			Map2 = sitk.ReadImage(inputPath + i) # middle PE map
			Map2_array = sitk.GetArrayFromImage(Map2)
		elif contrastID[2] in i:
			Map3 = sitk.ReadImage(inputPath + i) # late PE map
			Map3_array = sitk.GetArrayFromImage(Map3)
		else:
			print("Wrong match of MR images")
	ROI_Slice1 = getSlice(Map1_array)
	ROI_Slice2 = getSlice(Map2_array)
	ROI_Slice3 = getSlice(Map3_array)
	ROI_Slice_min = np.min([np.min(ROI_Slice1),np.min(ROI_Slice2),np.min(ROI_Slice3)])
	ROI_Slice_max = np.max([np.max(ROI_Slice1),np.max(ROI_Slice2),np.max(ROI_Slice3)])
	ROI_Slice = list(range(ROI_Slice_min,ROI_Slice_max + 1)) # z axis
	# early PE map
	Map1_YXindex = getYXindex(inputArray=Map1_array,inputSlice=ROI_Slice)
	# middle PE map
	Map2_YXindex = getYXindex(inputArray=Map2_array,inputSlice=ROI_Slice)
	# late PE map
	Map3_YXindex = getYXindex(inputArray=Map3_array,inputSlice=ROI_Slice)
	#
	M_YXindex = mergeYXindex(M1YXindex=Map1_YXindex,M2YXindex=Map2_YXindex,M3YXindex=Map3_YXindex)
	# early PE map, voxel value
	Lesion3D1 = getLesionRegion(inputArray=Map1_array,inputSlice=ROI_Slice,inputYXindex=M_YXindex)
	# middle PE map, voxel value
	Lesion3D2 = getLesionRegion(inputArray=Map2_array,inputSlice=ROI_Slice,inputYXindex=M_YXindex)
	# late PE map, voxel value
	Lesion3D3 = getLesionRegion(inputArray=Map3_array,inputSlice=ROI_Slice,inputYXindex=M_YXindex)
	# RGB array
	for j in range(0,len(ROI_Slice)):
		rgbArray = np.array([Lesion3D1[j], Lesion3D2[j], Lesion3D3[j]])
		# pad to 3*224*224
		padarray = getPadarray(inputRGB3Darray=rgbArray, TFXsize=224, TFYsize=224)
		padarray = padarray.astype(int)
		rgbImage = np.transpose(padarray, (1, 2, 0))  # from (z,y,x) to (x,y,z)
		patientRGB.append(rgbImage)
	return(patientRGB)


# datasets analysis P5
BRCA_MRIs_Path = '/Path/To/P4outputs/'
BRCA_mriFile = loadfiles(filePath = BRCA_MRIs_Path)
#
BRCA_outPath = '/Path/To/P5outputs/'
#
BRCA_pIDs = getpID(BRCA_mriFile)
contrastID = ['pre','post_1','post_5']
for i in range(0, len(BRCA_pIDs)):
	patientImageList = []
	for j in BRCA_mriFile:
		if BRCA_pIDs[i] in j:
			patientImageList.append(j)
	# get the RGB images
	Results = getRGB(inputImage = patientImageList, contrastID = contrastID, inputPath = BRCA_MRIs_Path)
	for k in range(0, len(Results)):
		filename = BRCA_outPath + BRCA_pIDs[i] + '-ValSca-N4-3Dcc-RGB_Slice_' + '' + str(k+1).rjust(3,'0') + '.jpg'
		io.imsave(filename, Results[k])

