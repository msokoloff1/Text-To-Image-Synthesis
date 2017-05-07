#Generator Class
# * Takes noise and a word embedding as input

from Network import Network
from textManager import *
import tensorflow as tf

class Generator(Network):
	def __init__(self, batchSize):
		self.textManager = TextModel(200, False, batchSize)
		self.batchSize = batchSize
		self.noisePH = tf.placeholder(tf.float32, shape = [batchSize, 100], name = "generatorNoisePH")
		with tf.variable_scope("GeneratorVars"):
			self.net = self._createNetwork()


	def _createNetwork(self):
		

		input = tf.concat([self.noisePH, self.textManager.trueOutput], axis = 1)
		reshaped = tf.reshape(input, [self.batchSize,1,1,300])
		layer1 = self._deconvLayer(inputLayer = reshaped
					  , filterWH = 5
					  , numOutputFeatures =  250
					  , stride =  2
					  , batchSize = self.batchSize 
					  , name = "dconv1")
		layer2 = self._deconvLayer(inputLayer = layer1
                                          , filterWH = 5
                                          , numOutputFeatures =  220
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv2")

		layer3 = self._deconvLayer(inputLayer = layer2
                                          , filterWH = 5
                                          , numOutputFeatures =  80
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv3")

		layer4 = self._deconvLayer(inputLayer = layer3
                                          , filterWH = 5
                                          , numOutputFeatures =  30
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv4")


		layer5 = self._deconvLayer(inputLayer = layer4
                                          , filterWH = 5
                                          , numOutputFeatures =  10
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv5")


		layer6 = self._deconvLayer(inputLayer = layer5
                                          , filterWH = 5
                                          , numOutputFeatures =  3
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv6"
					  , activation = tf.nn.tanh
					  , norm = False)
		self.output = (layer6/2.) + 0.5




