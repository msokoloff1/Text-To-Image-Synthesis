
#Program Entry Point
from Generator import *
import tensorflow as tf
from Discriminator import *
from Aggregator import *
from DataPrep import *


#Add support for altering images. (ie flip image. etc..)
#All ops are for 3d tensors, so something like this has to be used..
#result = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
#Add leaky relu
with tf.Session() as sess:
	batchSize = 64
	numIters = 500
	gen = Generator(batchSize)
	discrim = Discriminator(batchSize, gen)
	a = Aggregator(sess, discrim)
	saver = tf.train.Saver()
	try:
		saver.restore(sess, "savedModel.ckpt")
		print("Successfully Restored Model!!")
	except:
		sess.run(tf.global_variables_initializer())
		print("No model available for restoration")

	allData = loadAllData()
	
	a.learn(allData, numIters, batchSize)
	saver.save(sess, "savedModel.ckpt")
	
	
