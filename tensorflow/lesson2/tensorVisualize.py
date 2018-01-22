import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x  =tf.add(a,b)
with tf.Session() as sess:
#add this line to use Tensorboard
  writer = tf.summary.FileWriter('./graphs',sess.graph)
  print(sess.run(x))
  print(sess.graph.as_graph_def())
writer.close()
