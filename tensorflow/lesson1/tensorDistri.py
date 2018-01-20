import tensorflow as tf
#Creates a graph
with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a', shape=[1,6])
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b', shape=[6,1])
	c = tf.matmul(a,b)
# Creates a session with log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#Runs the op.
c = sess.run(c)
print(c)
