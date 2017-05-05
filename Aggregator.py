from Generator import *
from Discriminator import *

class Aggregator():
	def __init__(self,sess, generator, Discriminator):
		self.gen = generator
		self.discrim = Discriminator
		self.Sr, self.Sw, self.Sf = self.discrim.getOutputs()	
		self.updateDescrim = self._getUpdateDiscriminatorOp(0.0005)
		self.updateGen = self._getUpdateGeneratorOp(0.0005) 
	
	def _getUpdateGeneratorOp(self, learningRate):
		genLoss = tf.log(self.Sf) 
		optmizier = tf.train.AdamOptimizer(learningRate)
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "GeneratorVars")
		grads = optmizier.compute_gradients(genLoss, vars)
		return optimizer.apply_gradients(grads)
		

	def _getUpdateDiscriminatorOp(self, learningRate):
		DiscrimLoss = tf.log(self.Sr) + (tf.log(1.-self.Sw) + tf.log(1.-self.Sf))/2. 
		optmizier = tf.train.AdamOptimizer(learningRate)
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "GeneratorVars")
		grads = optmizier.compute_gradients(genLoss, vars)
		return optimizer.apply_gradients(grads)

	def updateGen(self, **feedDict):
		sess.run(self.updateGen, feed_dict = {})


	def updateDiscrim(self, **feedDict):
		sess.run(self..updateDescrim, feed_dict = {})
