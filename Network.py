#Contains functions to be inherited by the networks
import numpy as np
import tensorflow as tf


def leakyRelu(x):
	return tf.maximum(0.2*x, x)

class Network():

	def _convLayer(self, inputLayer, filterWH, numOutputFeatures, strideSize, name, activation= leakyRelu, padType = 'VALID'):
		numInputFeatures = inputLayer.get_shape().as_list()[-1]    
		convFilter = tf.Variable(tf.random_normal([filterWH, filterWH, numInputFeatures, numOutputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
		strideConfig =  [1, strideSize, strideSize, 1]
                
		
		
		conv = tf.nn.conv2d(inputLayer, convFilter, strideConfig, padding=padType)
		bias = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
		convOutput = tf.nn.bias_add(conv, bias)

		if activation == leakyRelu:
			norm = self._normalizeBatch(convOutput)
			return activation(norm)
		else:
			return activation(convOutput)

	def _residualBlock(self, inputLayer, name, filterWH=3):
		numInputFeatures = inputLayer.get_shape().as_list()[-1]
		conv = self._convLayer(inputLayer, filterWH, numInputFeatures, 1, "%s-a" % name)
		return inputLayer + self._normalizeBatch(self._convLayer(conv, filterWH, numInputFeatures, 1, name, activation=lambda x: x))

	def _normalizeBatch(self, x):
		with tf.variable_scope("instance_norm"):
			eps = 1e-6
			mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
			return (x - mean) / (tf.sqrt(var) + eps)

	def _deconvLayer(self, inputLayer, filterWH, numOutputFeatures, stride,batchSize, name, activation = leakyRelu, norm = True):
		_, h, w, numInputFeatures = inputLayer.get_shape().as_list()
            
		convFilter = tf.Variable(tf.random_normal([filterWH, filterWH,numOutputFeatures,numInputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
		deconv = tf.nn.conv2d_transpose(inputLayer, convFilter, [batchSize, h * stride, w * stride, numOutputFeatures], [1, stride, stride, 1], padding='SAME')
		biases = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
		bias = tf.nn.bias_add(deconv, biases)
		if(norm):
			return activation(self._normalizeBatch(bias))

		return activation(bias) 

