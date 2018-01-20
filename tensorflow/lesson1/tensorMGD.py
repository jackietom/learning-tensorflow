import tensorflow as tf
#Do not mix default graph and user created graphs
g1 = tf.get_default_graph()
g2 = tf.Graph()
#add ops to the default graph
with g1.as_default():
	a = tf.constant(3)
	
# add ops to the user created graph
with g2.as_default():
	b = tf.constant(5)
	
with tf.Session(graph=g1) as sess:
	x = sess.run(a)
	print(x)

with tf.Session(graph=g2) as sess:
	x = sess.run(b)
	print(x)
