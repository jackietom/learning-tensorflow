#### Tensorflow学习笔记 2018.1.22

##### 本讲内容

1. Tensorflow 中的线性回归算法
2. 优化器(Optimizers)
3. 在MNIST上进行logistic regression
4. 损失函数(loss function)

##### 一、线性回归

定义：对两个变量之间的关系进行建模

举例：X：火灾事故的数量，Y：盗窃事故的数量 目标：使用X去预测Y

模型：$Y = w*X + b, (Y - Y_predicted)^2$

编程过程：

1. 构建计算用图

   第一步：将数据读入

   第二步：为输入(inputs和labels)创建相应的placeholder

   ```
   tf.placeholder(dtype, shape=None, name=None)
   ```

   第三步：为权重(weight)和偏置(bias)声明变量(variable)

   ```
   tf.Variable(initial_value=None, trainable=True, collections=None, name=None, dtype=None, ...)
   ```

   第四步：构建用来预测Y的模型

   ```
   Y_predicted = X*w+b
   ```

   第五步：指定损失函数(loss function)

   ```
   tf.square(Y - Y_predicted, name="loss")
   ```

   第六步：创建优化器（用来调整权值和偏置）

   ```
   tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
   ```

2. 训练模型

   * 初始化变量
   * 将输入与label通过字典赋给placeholder,运行优化操作(run optimizier op)

3. 在Tensorboard中查看自己的模型

   第一步：

   ```
   Step 1: writer = tf.summary.FileWriter('./my_graph/03/linear_reg',sess.graph)
   ```

   第二步：

   ```
   $ tensorboard --logdir='./my_graph'
   ```

4. 使用matplotlib绘制出结果

   第一步：

   将程序底部的绘制语句的注释取消掉

   第二步：
   重新运行程序

解释：w, b = ses.run([w,b])会出现valueError错误

Tensorflow怎么知道更新哪一个变量？

通过优化器来实现：

```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
_, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
```

Session会寻找所有loss函数依靠的可训练（trainable)的变量并对它们进行更新

什么样的变量是可训练的？

```
tf.Variable(initial_value=None, trainable=True, collections=None,
validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None,
expected_shape=None, import_scope=None)
```

tensorflow中都有哪些优化器？

```
tf.train.GradientDescentOptimizer
tf.train.AdagradOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradOptimizer
tf.train.RMSPropOptimizer
And more
```

讨论问题：

1. 如何知道我们的模型是对的？

2. 怎么样提高我们的模型？

   通过改变loss函数来实现：Huber loss

   Huber loss的基本思想：

   当偏差不大时直接进行取平方，当偏差大于阈值时取其绝对值

   Huber loss的实现：

   ```python
   def huber_loss(labels, predictions, delta=1.0):
     residual = tf.abs(predictions - labels)
     condition = tf.less(residual, delta)
     small_res = 0.5 * tf.square(residual)
     large_res = delta * residual - 0.5*tf.square(delta)
     return tf.select(condition, small_res, large_res)
   ```

##### 二、logistic regression

MNIST数据集：

手写数字图片，大小为28x28，展平成一维tensor后长度为784

X:手写数字的图片 Y:数字的值 目标：使用X得到Y

模型：

```
logits = X*w + b
Y_predicted = softmax(logits)
loss = cross_entropy(Y, Y_predicted)
```

将数据集分为“批”

```
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")
```

处理数据：

从tensorflow.examples.tutorials.mnist 中导入输入数据

```
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

MNist.train: 55,000 examples
MNIST.validation: 5,000 examples
MNIST.test: 10,000 examples
```

1. 构建计算用图：

   第一步：

   为输入(inputs)和标志(labels)创建placeholder

   ```
   X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
   Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")
   ```

   第二步：创建权重和偏置变量

   ```
   tf.Variable(initial_value=None, trainable=True, collections=None,
   name=None, dtype=None, ...)
   ```

   第三步：构建用来预测的模型

   ```
   logits = X*w + b
   ```

   第四步：指定损失函数

   ```
   entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
   loss = tf.reduce_mean(entropy)
   ```

   第五步：创建优化器

   ```
   tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
   ```

2. 训练模型

   * 初始化变量
   * 运行优化操作

参考文献：

stanford cs20si_2017_slides_03

https://web.stanford.edu/class/cs20si/2017/lectures/slides_03.pdf

