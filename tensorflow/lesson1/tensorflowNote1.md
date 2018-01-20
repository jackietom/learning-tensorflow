#### Tensorflow 学习笔记

2018-1-20

##### 一、安装tensorflow 

​	经过若干次惨痛的经历，决定这次学习以ubuntu16.04 LTS虚拟机为编程环境进行测试。

由于demo例子体量都比较小，为了减省配置环境的不必要的时间浪费使用CPU版本的tensorflow。

安装命令：
​	sudo pip3 install tensorflow

##### 二、graph与session的定义

​	tensorflow是将最终的计算与网络的结构分开的，网络的结构就是graph

​	tensorflow的正确使用方法就是首先搭建一个网络，再调用一个session去执行。

什么是tensor?

​	tensor是张量，什么是张量我也不太清楚。但是我知道张量大概和矢量的感觉差不多，0维的tensor就是一个数，1维的tensor就是向量vector，2维的tensor就是矩阵matrix。

如何搭建网络？

​	搭建网络就是构建数据流图(data flow graphs)的过程。数据流图由节点和边组成。

节点(node):包含操作(operator),变量(variable),和常数(constants)

边(edge):边就是数据流，也就是tensor的传输

而节点中存储的也是tensor

示例代码：

​	import tensorflow as tf 

​	a = tf.add(3,5)

​	print(a)

此代码的输出并不是8，而是一个tensor的属性信息

在上述代码中并不能获得tensor的值，而如果要获取tensor的值的话就需要进行计算，也就是说需要使用CPU或GPU对其进行运算，这样就需要再程序和运算芯片之间建立连接，这个连接就叫做session(会话)

```python
import tensorflow as tf
a = tf.add(3,5)
sess = tf.Session()
print(sess.run(a))
sess.close()
```

这样就能够输出最终的计算结果8了

为了避免声明session之后忘记close可以采用python的with语句来自动关闭会话：

```python
import tensorflow as tf
a = tf.add(3,5)
with tf.Session() as sess:
	print(sess.run(a))
```

注：tf.Session()函数返回的session对象涵盖了对图中的计算进行执行和获取tensor的值的功能

更多的图像搭建（以下部分没有实质性的新知识，纯属练习性质）

“更多的图”：

```python
import tensorflow as tf
x = 2
y = 3
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
op3 = tf.pow(op2,op1)
with tf.Session() as sess:
  op3 = sess.run(op3)
  print(op3)
```

"亚图(subgraphs)"

```python
import tensorflow as tf
x = 2
y = 3
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)
useless = tf.multiply(x,add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
  z = sess.run(pow_op)
  print(z)
```

sess.run()函数可以同时计算多个tensor并返回：

```
import tensorflow as tf
x = 2
y = 3
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)
useless = tf.multiply(x,add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
  z,not_useless = sess.run([pow_op, useless])
  print(z)
  print(use_less)
```

##### 三、分布式计算

由于总图可以分为若干个亚图，故可以将整体的计算划分为不同的模块进行分别计算。

可以将一部分图放到CPU/GPU上进行运算：

代码示例：

```
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

```
can't map device in virtual mechine
注：不建议使用多个图，原因如下：

1. 多个图需要用到多个session，而每个session都默认会去使用所有的资源。
2. session之间不能传递数据，因为numpy不支持并行计算
3. 在同一个图中最好将不同的子图分开

##### 四、使用多个图

主要通过tf.Graph()函数来实现

简单的示例：

```python
import tensorflow as tf
#to add operators to a graph, set it as default:
g = tf.Graph()
with g.as_default():
	x = tf.add(3,5)
    
sess = tf.Session(graph=g)
#with tf.Session() as sess:
  x = sess.run(x)
  print(x)  
```

获取当前图的方法：

```
g = tf.get_default_graph()
```

当前默认图与用户创建的图是不一样的。（其实本质上是一样的，都是图。只是一个被指定为默认的结构图，而是否为默认的图还是可以切换的）

防混淆练习：

```
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
```

注：为什么使用“图”？

1. 节省计算量：通过对子图的复用来达到目的。
2. 通过将计算分为小的，不同的计算从而方便自动求导？（这一点没弄明白）
3. 可以用子图促进并行计算，从而利用GPU来进行加速
4. 许多机器学习的模型都已经被用图来进行了表示，方便代码实现



##### 参考文献

​	stanford CS20si_2017 lesson one:

https://web.stanford.edu/class/cs20si/2017/lectures/slides_01.pdf
