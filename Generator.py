#Generator Class
# * Takes noise and a word embedding as input

from Network import Network
from textManager import *
import tensorflow as tf

class Generator(Network):
	def __init__(self, textManager, batchSize):
		self.batchSize = batchSize
		self.textManager = textManager
		self.noisePH = tf.placeholder(tf.float32, shape = [None, 100])
		with tf.variable_scope("GeneratorVars"):
			self.net = self._createNetwork()

	def _createNetwork(self):


		input = tf.concat([self.noisePH, self.textManager.getOutputEmbedding()], axis = 1)
		print(input.get_shape())
		reshaped = tf.reshape(input, [self.batchSize,1,1,300])
		print(reshaped.get_shape())
		layer1 = self._deconvLayer(inputLayer = reshaped
					  , filterWH = 5
					  , numOutputFeatures =  250
					  , stride =  2
					  , batchSize = self.batchSize 
					  , name = "dconv1")
		print(layer1.get_shape())
		layer2 = self._deconvLayer(inputLayer = layer1
                                          , filterWH = 5
                                          , numOutputFeatures =  220
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv2")
		print(layer2.get_shape())

		layer3 = self._deconvLayer(inputLayer = layer2
                                          , filterWH = 5
                                          , numOutputFeatures =  80
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv3")
		print(layer3.get_shape())

		layer4 = self._deconvLayer(inputLayer = layer3
                                          , filterWH = 5
                                          , numOutputFeatures =  30
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv4")
		print(layer4.get_shape())


		layer5 = self._deconvLayer(inputLayer = layer4
                                          , filterWH = 5
                                          , numOutputFeatures =  10
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv5")
		print(layer5.get_shape())


		layer6 = self._deconvLayer(inputLayer = layer5
                                          , filterWH = 5
                                          , numOutputFeatures =  3
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv6"
					  , activation = tf.nn.tanh
					  , norm = False)
		print(layer6.get_shape())
		self.output = (layer6/2.) + 0.5


model = TextModel(200)
gen = Generator(model,1)


