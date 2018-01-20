import tensorflow as tf
#to add operators to a graph, set it as default:
g = tf.Graph()
with g.as_default():
	x = tf.add(3,5)
    
sess = tf.Session(graph=g)
#with tf.Session() as sess:
x = sess.run(x)
print(x)  
