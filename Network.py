#Contains functions to be inherited by the networks
import numpy as np
import tensorflow as tf
class Network():
	def _residualBlock(self, inputLayer, name, filterWH=3):
		with tf.variable_scope(name):
			numInputFeatures = inputLayer.get_shape().as_list()[-1]
			conv = self.__convLayer__(inputLayer, filterWH, numInputFeatures, 1, "%s-a" % name)
			return inputLayer + self.__normalizeBatch__(self.__convLayer__(conv, filterWH, numInputFeatures, 1, name, activation=lambda x: x))

	def _normalizeBatch(self,x):
		with tf.variable_scope("instance_norm"):
			eps = 1e-6
			mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
			return (x - mean) / (tf.sqrt(var) + eps)


	def _deconvLayer(self, inputLayer, filterWH, numOutputFeatures, stride,batchSize, name):
		with tf.variable_scope(name):
			_, h, w, numInputFeatures = inputLayer.get_shape().as_list()
            
			convFilter = tf.Variable(tf.random_normal([filterWH, filterWH,numOutputFeatures,numInputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
			deconv = tf.nn.conv2d_transpose(inputLayer, convFilter, [batchSize, h * stride, w * stride, numOutputFeatures], [1, stride, stride, 1], padding='SAME')
			biases = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
			bias = tf.nn.bias_add(deconv, biases)
			norm = self._normalizeBatch(bias)
			return tf.nn.relu(norm)
        

