from Generator import *
from Discriminator import *
import numpy as np
from random import randint

class Aggregator():
	def __init__(self,sess, discriminator):
		self.discrim = discriminator
		self.gen = self.discrim.generator
		self.Sr, self.Sw, self.Sf = self.discrim.getOutputs()	
		
		self.sess = sess
		self.updateDiscrim = self._getUpdateDiscriminatorOp(0.00002)
		self.updateGen = self._getUpdateGeneratorOp(0.000002) 
	
	def _getUpdateGeneratorOp(self, learningRate):
		genLoss = -tf.reduce_mean(tf.log(self.Sf+0.0001))
		optimizer = tf.train.AdamOptimizer(learningRate)
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "GeneratorVars")
		grads = optimizer.compute_gradients(genLoss, vars)
		tf.summary.scalar("genLoss", genLoss)
		tf.Print(genLoss, [genLoss], message = "Generator Loss : ")
		return optimizer.apply_gradients(grads)
		

	def _getUpdateDiscriminatorOp(self, learningRate):
		DiscrimLoss = -tf.reduce_mean(tf.log(self.Sr+ 0.0001) + (tf.log(1.00001-self.Sw) + tf.log(1.0001-self.Sf))/2.) 
		tf.summary.scalar("discrimLoss", DiscrimLoss)
		optimizer = tf.train.AdamOptimizer(learningRate)
		vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "discriminatorVars")
		grads = optimizer.compute_gradients(DiscrimLoss, vars)
		tf.Print(DiscrimLoss, [DiscrimLoss], message = "Discriminator Loss : ")
		return optimizer.apply_gradients(grads)

	def _applyGenUpdate(self,sentences, batchSize):
		trueEmbeddings = self._prepSentences(sentences, batchSize, self.gen.textManager)

		noise = np.random.random((batchSize,100))
		self.sess.run(self.updateGen,  feed_dict = {
			  self.gen.noisePH:noise
			, self.gen.textManager.trueSentencePlaceholder:trueEmbeddings
			, self.discrim.textManager.trueSentencePlaceholder:trueEmbeddings
			})

	def _prepSentences(self, sentences, batchSize, embeddingNetwork):
		embeddings = np.zeros((batchSize,embeddingNetwork.seqLen, embeddingNetwork.outputDim)) 
		for index, sentence in enumerate(sentences):
			embeddings[index] = embeddingNetwork.getSentenceEmbedding(sentence)

		return embeddings



	def _applyDiscrimUpdate(self, falseSentences, trueSentences,realImages, batchSize, iteration,tb = False):
		trueEmbeddings = self._prepSentences(trueSentences, batchSize, self.discrim.textManager)
		fakeEmbeddings = self._prepSentences(falseSentences, batchSize, self.discrim.textManager)
		noise = np.random.random((batchSize,100))
		feedDict = {
 			  self.gen.noisePH:noise
                        , self.gen.textManager.trueSentencePlaceholder:trueEmbeddings
                        , self.discrim.textManager.trueSentencePlaceholder:trueEmbeddings
                        , self.discrim.imageReal:realImages
                        , self.discrim.textManager.falseSentencePlaceholder:fakeEmbeddings
                        }
		
		if(tb):
			summary,_ = self.sess.run([self.summary_op,self.updateDiscrim], feed_dict=feedDict)
			self.write.add_summary(summary, iteration)
		else:
			self.sess.run(self.updateDiscrim, feed_dict=feedDict)
		

	def learn(self, allData, numIters, batchSize):
		self.summary_op = tf.summary.merge_all()
		self.write = tf.summary.FileWriter("tboard")
		dKeys = list(allData.keys())
		for iteration in range(numIters):
			once = True
			for index in range(0,len(dKeys), batchSize):
				images = np.random.random( (batchSize, 64,64,3) )
				trueText, falseText = [],[]
				for batchIndex in range(batchSize):
					images[batchIndex] = allData[dKeys[index+batchIndex-(batchSize+1)]]['image']
					trueText.append(allData[dKeys[index+batchIndex-(batchSize+1)]]['text'][randint(0,4)])
					falseText.append(allData[dKeys[randint(0,len(dKeys)-1)]]['text'][randint(0,4)])
				self._applyDiscrimUpdate(falseText, trueText, images, batchSize, iteration, once)
				once = False
				self._applyGenUpdate(trueText, batchSize)
					


