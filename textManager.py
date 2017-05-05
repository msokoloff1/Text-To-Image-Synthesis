import gensim
import string
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
#Gensim will map each word to an embedding
#Each embedding becomes an input at a particular time step for an rnn
#The rnn output will produce the final text embedding output
#The alternative approach will be the output of the embeddings are just a fully connected layer
class TextModel():
	def __init__(self,oneDimLen, seqLen = 30, fc = True):
		#self.model = gensim.models.KeyedVectors.load_word2vec_format('./resources/textEmbeddings.bin', binary=True)
		self.model = gensim.models.KeyedVectors.load('./resources/small.bin')
		self.seqLen = seqLen
		self.is_fc = fc
		self.outputDim = oneDimLen
		self.sentencePlaceholder = tf.placeholder(tf.float32, shape = [None,seqLen,300])
		with tf.variable_scope("embedding_vars") as scope:
			if(fc):
				self._output = self._fcLayer()
			else:
				self._output = self._rnnLayer()


	def getSentenceEmbedding(self, sentence):
		tbl = str.maketrans({key: None for key in string.punctuation})
		tokenizedSentence = sentence.translate(tbl).split(" ")
		filtered_words = [word for word in tokenizedSentence if word not in stopwords.words('english')]
		if(len(filtered_words) > self.seqLen):
			filtered_words = filtered_words[:self.seqLen] 

		matrixRep = np.zeros((self.seqLen, 300))

		for index, token in enumerate(filtered_words):

			try:
				matrixRep[index] = self.model.word_vec(token)
			except:
				matrixRep[index] = np.ones(300)
			
		return matrixRep

	def _rnnLayer(self):
		pass

	def _fcLayer(self):
		weights = tf.Variable(tf.truncated_normal([self.seqLen * 300 , self.outputDim], stddev=0.1))
		bias = tf.Variable(tf.constant(1000., shape = [self.outputDim]))
		reshaped = tf.reshape(self.sentencePlaceholder, [-1,self.seqLen * 300]) 
		return tf.nn.relu(tf.matmul(reshaped, weights) + bias)

	def getOutputEmbedding(self):
		return self._output

