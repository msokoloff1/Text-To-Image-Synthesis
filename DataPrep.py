#This file handles text and image input
from scipy import misc
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
imagePath = "./resources/jpg"

def cleanDir():
	dest = "./resources/labels/allLabels"
	if(not os.path.exists(dest)):
		os.makedirs(dest)
		buildRoot = "./resources/labels/text_c10"
		for fold in os.listdir(buildRoot):
			for fileNames in os.listdir(buildRoot + "/" + fold):
				if(fileNames.endswith('.txt')):	
					source = buildRoot+'/'+fold+'/'+fileNames
					os.rename(source, dest+ "/" + fileNames)
						

datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False, 
        zca_whitening=False,
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)


def loadImage(path, imageShape = (64,64,3)):
	subjectImage = misc.imresize(misc.imread(path), imageShape) / 255.
	return datagen.flow(subjectImage.reshape(1,64,64,3),batch_size = 1)

def getText(imageName):
	filePath = "./resources/labels/allLabels/" +imageName + ".txt"
	with open(filePath,'r') as file: 
		return file.read().split('\n')[0:5]
	


def loadAllData():
	cleanDir()
	jpgPath = "./resources/jpg"
	data = {}
	for imageName in os.listdir(jpgPath):
		data[imageName[:-4]] = {
			'image' : loadImage(jpgPath + "/"+ imageName)
			 , 'text' : getText(imageName[:-4])
		}		

	return data



