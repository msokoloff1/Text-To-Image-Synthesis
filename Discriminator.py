from Network import Network
import numpy as np
import tensorflow as tf
from textManager import *

class Discriminator(Network):
	def __init__(self, batchSize, textManager):
		#This text manager must be different than the generators.
		self.inputCorrect = tf.placeholder(tf.float32, shape = [batchSize,64,64,3]) 
		self.inputWrong = tf.placeholder(tf.float32, shape = [batchSize,64,64,3])
		self.textManager = textManager
		self._createNetwork()
		#Get multiple text input tensors... need reuse variable in textManager.
				
	def _createNetwork(self):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		#Try residual blocks in the future
		def p(l):
			mult = 1.
			for x in l:
				mult*=float(int(x))
			print(mult)

		layer1 = self._convLayer(self.input, 3, 10, 2, "conv1")
		p(layer1.get_shape())
		layer2 = self._convLayer(layer1, 3, 30, 2, "conv2")
		p(layer2.get_shape())
		layer3 = self._convLayer(layer2, 3, 90, 2, "conv3")
		p(layer3.get_shape())
		layer4 = self._convLayer(layer3, 3, 120, 2, "conv4", padType = "SAME")
		p(layer4.get_shape())
		reshaped = tf.reshape(self.textManager.getOutputEmbedding(), [1,1,1,200])	
		tiled = tf.tile(reshaped, [1,4,4,1])
		print(tiled.get_shape())
		layer4Concated = tf.concat([layer4, tiled], axis = 3)
		layer5 = self._convLayer(layer4Concated,1,30,1, "conv5")
		p(layer5.get_shape())
		outputLayer = self._convLayer(layer5, 4,1,1,"d_output",activation = tf.nn.tanh)
		self.output = ((outputLayer/2) + 0.5)
		
	def   
#Use convolution until shape is 4x4. Then a 1x1 conv, with tanh mapped between 0 -1 
model = TextModel(200)
d = Discriminator(1, model)	

