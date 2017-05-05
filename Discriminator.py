from Network import Network
import numpy as np
import tensorflow as tf
from textManager import *

class Discriminator(Network):
	def __init__(self, batchSize, generatorOutputTensor):
		
		#This text manager must be different than the generators.
		self.TextModel = TextModel(200, True, batchSize)
		self.imageReal = tf.placeholder(tf.float32, shape = [batchSize,64,64,3]) 
		self.imageGenerated = generatorOutputTensor
		self.textManager = textManager
				
	def getOutputs(self):
		Sr = self._discriminate(self.imageReal, self.TextModel.trueSentencePlaceholder, reuse = False)
		Sw = self._discriminate(self.imageReal, self.TextModel.falseSentencePlaceholder)
		Sf = self._discriminate(self.imageGenerated, self.TextModel.trueSentencePlaceholder)
		return Sr, Sw, Sf 

	def _discriminate(self, imageTensor,textTensor, reuse=True):

		if reuse:
			tf.get_variable_scope().reuse_variables()

		layer1 = self._convLayer(imageTensor, 3, 10, 2, "conv1")
		layer2 = self._convLayer(layer1, 3, 30, 2, "conv2")
		layer3 = self._convLayer(layer2, 3, 90, 2, "conv3")
		layer4 = self._convLayer(layer3, 3, 120, 2, "conv4", padType = "SAME")
		reshaped = tf.reshape(textTensor, [1,1,1,200])	
		tiled = tf.tile(reshaped, [1,4,4,1])
		layer4Concated = tf.concat([layer4, tiled], axis = 3)
		layer5 = self._convLayer(layer4Concated,1,30,1, "conv5")
		outputLayer = self._convLayer(layer5, 4,1,1,"d_output",activation = tf.nn.tanh)
		self.output = ((outputLayer/2) + 0.5)
		

d = Discriminator(1)	

