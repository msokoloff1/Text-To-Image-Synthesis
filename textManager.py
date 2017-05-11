import gensim
import string
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
#Gensim will map each word to an embedding
#Each embedding becomes an input at a particular time step for an rnn
#The rnn output will produce the final text embedding output
#The alternative approach will be the output of the embeddings are just a fully connected layer
#http://stackoverflow.com/documentation/tensorflow/4827/creating-rnn-lstm-and-bidirectional-rnn-lstms-with-tensorflow#t=201705082127298191823
	
	
class TextModel():
	def __init__(self,oneDimLen,discriminator,batchSize,seqLen = 30, fc = True):
		self.model = gensim.models.KeyedVectors.load_word2vec_format('./resources/textEmbeddings.bin', binary=True)
		#self.model = gensim.models.KeyedVectors.load('./resources/small.bin')
		self.seqLen = seqLen
		self.is_fc = fc
		self.outputDim = oneDimLen
		name = "generator"
		if(discriminator):
			name = "discriminator"

		self.trueSentencePlaceholder = tf.placeholder(tf.float32, shape = [batchSize,seqLen,self.outputDim], name = "trueSent"+name+"PH")


		with tf.variable_scope(name):	
			self.trueOutput = self._fcLayer(self.trueSentencePlaceholder, False)

			if(discriminator):
				self.falseSentencePlaceholder = tf.placeholder(tf.float32, shape = [batchSize,seqLen,self.outputDim], name = "falseSentPH")
				self.falseOutput = self._fcLayer(self.falseSentencePlaceholder,True)


	def getSentenceEmbedding(self, sentence):
		tbl = str.maketrans({key: None for key in string.punctuation})
		tokenizedSentence = sentence.translate(tbl).split(" ")
		filtered_words = [word for word in tokenizedSentence if word not in stopwords.words('english')]
		if(len(filtered_words) > self.seqLen):
			filtered_words = filtered_words[:self.seqLen] 

		matrixRep = np.zeros((self.seqLen, self.outputDim))

		for index, token in enumerate(filtered_words):

			try:
				matrixRep[index] = self.model.word_vec(token)
			except:
				matrixRep[index] = np.ones(self.outputDim)
			
		return matrixRep

	def _rnnLayer(self):
		pass

	def _fcLayer(self, placeholder, reuse):
		if(reuse):
			tf.get_variable_scope().reuse_variables()

		weights = tf.Variable(tf.truncated_normal([self.seqLen * self.outputDim , self.outputDim], stddev=0.1))
		bias = tf.Variable(tf.constant(.1, shape = [self.outputDim]))
		reshaped = tf.reshape(placeholder, [-1,self.seqLen * self.outputDim]) 
		return tf.nn.relu(tf.matmul(reshaped, weights) + bias)

