from Generator import *
from Discriminator import *
import numpy as np

class Aggregator():
	def __init__(self,sess, discriminator):
		self.discrim = discriminator
		self.gen = self.discrim.generator
		self.Sr, self.Sw, self.Sf = self.discrim.getOutputs()	
		self.sess = sess
		self.updateDescrim = self._getUpdateDiscriminatorOp(0.0005)
		self.updateGen = self._getUpdateGeneratorOp(0.0005) 
	
	def _getUpdateGeneratorOp(self, learningRate):
		genLoss = tf.log(self.Sf) 
		optimizer = tf.train.AdamOptimizer(learningRate)
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "GeneratorVars")
		grads = optimizer.compute_gradients(genLoss, vars)
		return optimizer.apply_gradients(grads)
		

	def _getUpdateDiscriminatorOp(self, learningRate):
		DiscrimLoss = tf.log(self.Sr) + (tf.log(1.-self.Sw) + tf.log(1.-self.Sf))/2. 
		optimizer = tf.train.AdamOptimizer(learningRate)
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "discriminatorVars")
		grads = optimizer.compute_gradients(DiscrimLoss, vars)
		return optimizer.apply_gradients(grads)

	def _applyGenUpdate(self,sentences, batchSize):
		trueEmbeddings = np.zeros((batchSize, self.gen.textManager.seqLen, self.gen.textManager.outputDim))
		for index, sentence in enumerate(sentences):
			trueEmbeddings[index] = self.gen.textManager.getSentenceEmbedding(sentence)
		

		noise = np.random.random((batchSize,100))
		self.sess.run(self.updateGen,  feed_dict = {
			  self.gen.noisePH:noise
			, self.gen.textManager.trueSentencePlaceholder:trueEmbeddings
			, self.discrim.textManager.trueSentencePlaceholder:trueEmbeddings
			})



	def _applyDiscrimUpdate(self):
		pass
		#sess.run(self..updateDescrim, feed_dict = {})

	def learn(self):
		pass


with tf.Session() as sess:
	batchSize = 2
	gen = Generator(batchSize)
	discrim = Discriminator(batchSize, gen) 
	a = Aggregator(sess, discrim)
	sess.run(tf.global_variables_initializer())
	a._applyGenUpdate(["the cat in the hat","the fat in the hat"],batchSize)
