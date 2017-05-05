#Contains functions to be inherited by the networks
import numpy as np
import tensorflow as tf
class Network():
	def _convLayer(self, inputLayer, filterWH, numOutputFeatures, strideSize, name, activation= tf.nn.relu, padType = 'VALID'):
		numInputFeatures = inputLayer.get_shape().as_list()[-1]    
		convFilter = tf.Variable(tf.random_normal([filterWH, filterWH, numInputFeatures, numOutputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
		strideConfig =  [1, strideSize, strideSize, 1]
                
		conv = tf.nn.conv2d(inputLayer, convFilter, strideConfig, padding=padType)
		bias = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
		convOutput = tf.nn.bias_add(conv, bias)

		if activation == tf.nn.relu:
			norm = self._normalizeBatch(convOutput)
			return activation(norm)
		else:
			return activation(convOutput)

	def _residualBlock(self, inputLayer, name, filterWH=3):
		with tf.variable_scope(name):
			numInputFeatures = inputLayer.get_shape().as_list()[-1]
			conv = self._convLayer(inputLayer, filterWH, numInputFeatures, 1, "%s-a" % name)
			return inputLayer + self._normalizeBatch(self._convLayer(conv, filterWH, numInputFeatures, 1, name, activation=lambda x: x))

	#def _normalizeBatch(self,x):
	#	with tf.variable_scope("instance_norm"):
	#		eps = 1e-6
	#		mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
	#		return (x - mean) / (tf.sqrt(var) + eps)
	def _normalizeBatch(self, x):
		with tf.variable_scope("batch_norm"):
			mean = tf.Variable(tf.constant(0.1 ,shape = [1]))
			variance = tf.Variable(tf.constant(0.1, shape = [1]))
			beta = tf.Variable(tf.constant(0.1 ,shape = [1]))
			gamma = tf.Variable(tf.constant(0.1, shape = [1]))
			return tf.nn.batch_normalization(x,mean, variance,beta, gamma, 0.00001) 

	def _deconvLayer(self, inputLayer, filterWH, numOutputFeatures, stride,batchSize, name, activation = tf.nn.relu, norm = True):
		with tf.variable_scope(name):
			_, h, w, numInputFeatures = inputLayer.get_shape().as_list()
            
			convFilter = tf.Variable(tf.random_normal([filterWH, filterWH,numOutputFeatures,numInputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
			deconv = tf.nn.conv2d_transpose(inputLayer, convFilter, [batchSize, h * stride, w * stride, numOutputFeatures], [1, stride, stride, 1], padding='SAME')
			biases = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
			bias = tf.nn.bias_add(deconv, biases)
			if(norm):
				return activation(self._normalizeBatch(bias))

			return activation(bias) 

