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
		self.net = self._createNetwork()

	def _createNetwork(self):


		input = tf.concat([self.noisePH, self.textManager.getOutputEmbedding()], axis = 1)
		reshaped = tf.reshape(input, [self.batchSize,1,1,200])
		print(reshaped.get_shape())
		layer1 = self._deconvLayer(inputLayer = reshaped
					  , filterWH = 5
					  , numOutputFeatures =  256
					  , stride =  2
					  , batchSize = self.batchSize 
					  , name = "dconv1")
		print(layer1.get_shape())
		layer2 = self._deconvLayer(inputLayer = layer1
                                          , filterWH = 5
                                          , numOutputFeatures =  256
                                          , stride =  2
                                          , batchSize = self.batchSize
                                          , name = "dconv1")
		print(layer2.get_shape())
		#layer2 = self._deconvLayer()


model = TextModel(200)
gen = Generator(model,1)

