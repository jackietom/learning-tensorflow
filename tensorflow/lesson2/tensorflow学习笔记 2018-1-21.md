#### tensorflow学习笔记 2018-1-21

第二节课内容：

1. 基本操作
2. tensor的类型
3. 项目速度的标记
4. 占位符和相应的输入
5. lazy loading

##### 一、使用tensorboard将网络可视化

示例：

```python
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)
with tf.Session() as sess:
	#add this line to use Tensorboard
    writer = tf.summary.FileWriter('./graphs,sess.graph')
    print(sess.run(x))
writer.close()#close the writer when you're done using it 
```

打开终端：

```
$ python3 [yourprogram].py
$ tenserboard --logdir="./graphs" --port 6006
```

打开浏览器访问"http://localhost:6006/"即可



这时候显示出的节点使用默认的名称（如const, const_1等），要想显示为变量名则需要显式地为其赋名：

```
import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a,b,name="add")
writer = tf.summary.FileWriter("./graphs", sess.graph)
with tf.Session() as sess:
	print(sess.run(x)) # >> 5
```

也可直接输出图的定义：

```python
import tensorflow as tf
my_const = tf.constant([1.0,2.0], name="my_const")
with tf.Session() as sess:
  print(sess.graph.as_graph_def())
```

##### 二、constant详细介绍

语法：

```
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
```

举例：

```python
import tensorflow as tf
a = tf.constant([2,2],name="a")
b = tf.constant([[0,1],[2,3]], name="b")
x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="multiply")
with tf.Session() as sess:
	x,y = sess.run([x,y])
    print(x,y)
# >> [5 8] [6 12]
```

生成具有指定值的tensor:

1. tf.zeros(shape, dtype=tf.float32, name=None)

   tf.zeros([2,3], tf.int32) ==> [[0,0,0],[0,0,0]]

2. tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)

   input_tensor is [[0,1], [2,3], [4,5]]

   tf.zeros_like(input_tensor) ==> [[0,0],[0,0],[0,0]]

3. tf.ones(shape, dtype=tf.float32, name=None)

4. tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

5. tf.fill(dims, value, name=None)

   tf.fill([2,3],8) ==> [[8,8,8],[8,8,8]]

生成constants序列：

1. tf.linspace(start,stop,num,name=None)

   tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 3.0]

2. tf.range(start, limit=None, delta=1, dtype=None, name=None)

   'start':3 'limit':18 'delta':3

   tf.range(start,limit,delta) ==> [3,6,9,12,15]

   limit:5

   tf.range(limit) ==> [0, 1, 2, 3, 4]

生成随机constants序列（具体说明用到时再查文档吧）：

1. tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
2. tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
   name=None)
3. tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
   name=None)
4. tf.random_shuffle(value, seed=None, name=None)
5. tf.random_crop(value, size, seed=None, name=None)
6. tf.multinomial(logits, num_samples, seed=None, name=None)
7. tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

设置种子：tf.set_random_seed(seed)

##### 三、operation详细介绍

（1）总体分类：

1. Element-wise mathematical operations(对数进行一一对应的操作)

   Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal, ...

2. Array operations(对数组进行操作)

   Concat, Slice, Split, Constant, Rank, Shape, Shuffle, ...

3. Matrix operations(对矩阵进行操作)

   MatMul, MatrixInverse, MatrixDeterminant, ...

4. Stateful operations(声明性质的操作)

   Variable, Assign, AssignAdd, ...

5. Neural network building blocks(搭建神经网络的操作)

   SoftMax, Sigmoid, Relu, Convolution2D, MaxPool, ...

6. Checkpointing operations(存储网络结构的操作)

   Save,Restore

7. Queue and synchronization operations(队列和同步操作)

   Enqueue, Dequeue, MutexAcquire, MutexRelease, ...

8. Control flow operations(流控制操作)

   Merge, Switch, Enter, Leave, NextIteration

（2）示例：

```python
a = tf.constant([3, 6])
b = tf.constant([2, 2])
tf.add(a, b) # >> [5 8]
tf.add_n([a, b, b]) # >> [7 10]. Equivalent to a + b + b
tf.mul(a, b) # >> [6 12] because mul is element wise
tf.matmul(a, b) # >> ValueError
tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1])) # >> [[18]]
tf.div(a, b) # >> [1 3] 
tf.mod(a, b) # >> [1 0]
```

##### 四、Tensorflow 的数据类型

Tensorflow使用python自身的数据类型：boolean, numeric(int float), strings

示例：

```python
0-d tensor, or "scalar"
t_0 = 19
tf.zeros_like(t_0) # ==> 0
tf.ones_like(t_0) # ==> 1
1-d tensor, or "vector"
t_1 = ['apple', 'peach', 'banana']
tf.zeros_like(t_1) # ==> ['' '' '']
tf.ones_like(t_1) # ==> TypeError: Expected string, got 1 of type 'int' instead.
2x2 tensor, or "matrix"
t_2 = [[True, False, False],
 [False, False, True],
 [False, True, False]]
tf.zeros_like(t_2) # ==> 2x2 tensor, all elements are False
tf.ones_like(t_2) # ==> 2x2 tensor, all elements are True

```

Tensorflow与numpy数据类型的关系：

二者是无缝衔接的：

```python
tf.int32 == np.int32 # True
```

可以将numpy的数据类型传递给tensorflow:

```python
tf.ones([2, 2], np.float32) # ⇒ [[1.0 1.0], [1.0 1.0]]
```

对于：

```
tf.Session.run(fetches)
```

如果fetch是tensor，则输出的是一个Numpy ndarray

注：不能使用python自身的数据类型，因为tensorflow需要推断python的类型(?)

尽量不要直接使用numpy的array，因为tensorflow未来可能会与numpy不再兼容

Constants的问题：constants是定义在图的定义中的：

输出图的定义：

```python
import tensorflow as tf
my_const = tf.constant([1.0,2.0], name="my_const")
with tf.Session() as sess:
  print(sess.graph_as_graph_def())
```

这使得加载图的代价增大。

解决方案：只对基本的数据使用constants表示，对于更多的数据可以采用variable或者reader进行表示。但是这也会有一个问题，这会占用更多的内存。

 ##### 五、variable详细介绍

variable示例：

```python
# create variable a with scalar value
a = tf.Variable(2, name="scalar")
# create variable b as a vector
b = tf.Variable([2, 3], name="vector")
# create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]))
```

注意到variable的V是大写的，这与constants不同。原因是constants是声明的操作，而variable是一个类，它具有很多的方法：

```
x = tf.Variable(...)
x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more
```

使用variable的时候首先要对其进行初始化

可以一次性的初始化所有变量

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
```

也可以只初始化特定的variable

```python
init_ab = tf.variables_initializer([a,b], name="init_ab")
with tf.Session() as sess:
  sess.run(init_ab)
```

还可以只初始化一个variable

```python
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
  sess.run(W.initializer)
```

获取一个variable的值：

```python
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W.eval())
```

向variable赋值：

```python
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
  sess.run(W.initializer)
  print(W.eval()) # >> 10
  
---------------------------

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
  #sess.run(W.initializer)
  ses.run(assign_op)
print(W.eval()) # >> 100
```

assign_op自带初始化功能，不必显式初始化。而初始化操作的本质就是将初始值assign给变量。

可以迭代多次赋值：

```
# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var")
# assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
sess.run(my_var.initializer)
sess.run(my_var_times_two) # >> 4
sess.run(my_var_times_two) # >> 8
sess.run(my_var_times_two) # >> 16
```

assign_add()和assign_sub()操作：分别对变量进行加和减。

```
my_var = tf.Variable(10)
With tf.Session() as sess:
sess.run(my_var.initializer)
# increment by 10
sess.run(my_var.assign_add(10)) # >> 20
# decrement by 2
sess.run(my_var.assign_sub(2)) # >> 18
```

每一个Session对于变量都会有自己的一份版本

```
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print sess1.run(W.assign_add(10)) # >> 20
print sess2.run(W.assign_sub(2)) # >> 8

```

使用一个变量去初始化另一个变量的方法：

1. 直接初始化

   ```
   # W is a random 700 x 100 tensor
   W = tf.Variable(tf.truncated_normal([700, 10]))
   U = tf.Variable(2 * W)
   ```

   这种方法不是很安全但很常见

2. 保证用来进行初始化的变量已经被初始化

   ```
   # W is a random 700 x 100 tensor
   W = tf.Variable(tf.truncated_normal([700, 10]))
   U = tf.Variable(2 * W.intialized_value())
   ```

   上一种方法需要保证W先被初始化，而当前的方法可以保证这一点。

##### 六、Session, InteractiveSession, Control Dependencies

​	Session 和InteractiveSession的区别

​	二者在功能上是一样的，只是在使用InteractiveSession的时候，默认其自身为当前Session，好处就是不用显式地声明sess.run()

示例：

```
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without specifying the context 'sess'
print(c.eval())
sess.close()

```

​	Control dependencies:用来指定当前图中哪些操作需要在另一些操作已经执行的基础上再执行

示例：

```
# your graph g have 5 ops: a, b, c, d, e
with g.control_dependencies([a, b, c]):
# 'd' and 'e' will only run after 'a', 'b', and 'c' have executed.
d = ...
e = …

```

##### 七、Project Speed Dating（不知道怎么翻译）

1. Placeholder简介

   * 为什么使用placeholder？

     placeholder更像函数的自变量，尽管我们不一定知道它具体的值但是我们可以事先利用它来定义图的结构，然后再传给它值。

   * 语法

     ```
     tf.placeholder(dtype, shape=None, name=None)
     ```

     示例：

     ```
     # create a placeholder of type float 32-bit, shape is a vector of 3 elements
     a = tf.placeholder(tf.float32, shape=[3])
     # create a constant of type float 32-bit, shape is a vector of 3 elements
     b = tf.constant([5, 5, 5], tf.float32)
     # use the placeholder as you would a constant or a variable
     c = a + b # Short for tf.add(a, b)
     with tf.Session() as sess:
     print sess.run(c) # Error because a doesn’t have any value
     ```

   * 使用字典为placeholder赋值

     ```
     # create a placeholder of type float 32-bit, shape is a vector of 3 elements
     a = tf.placeholder(tf.float32, shape=[3])
     # create a constant of type float 32-bit, shape is a vector of 3 elements
     b = tf.constant([5, 5, 5], tf.float32)
     # use the placeholder as you would a constant or a variable
     c = a + b # Short for tf.add(a, b)
     with tf.Session() as sess:
     # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
     # fetch value of c
     print sess.run(c, {a: [1, 2, 3]}) # the tensor a is the key, not the string ‘a’
     # >> [6, 7, 8]

     ```

     注：shape=None意味着该placeholder可以接受任意形状的值，但是不利于debug

     这也可能让它后面对形状有要求的placeholder无法正常工作。而且placeholder也是一种操作(op)

   * 如何向placeholder中输入一系列的值？

     一次输入一个：

     ```
     with tf.Session() as sess:
     for a_value in list_of_values_for_a:
     print sess.run(c, {a: a_value})
     ```

     注：可以向placeholder中输入任何可以输入的tensor(feedable)

     ```
     tf.Graph.is_feedable(tensor)
     #True if and only if tensor is feedable
     ```

   * 向tensorflow的操作(op)中赋值

     ```
     # create operations, tensors, etc (using the default graph)
     a = tf.add(2, 5)
     b = tf.mul(a, 3)
     with tf.Session() as sess:
     # define a dictionary that says to replace the value of 'a' with 15
     replace_dict = {a: 15}
     # Run the session, passing in 'replace_dict' as the value to 'feed_dict'
     sess.run(b, feed_dict=replace_dict) # returns 45

     ```

     这对于测试是很有用的

##### 八、lazy loading

1. 什么是lazy loading?

   仅在需要的时候创建或初始化一个对象

   示例：

   正常的loading

   ```
   x = tf.Variable(10, name='x')
   y = tf.Variable(20, name='y')
   z = tf.add(x, y) # you create the node for add node before executing the graph
   with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
   for _ in range(10):
   sess.run(z)
   writer.close()

   ```

   add操作在图中会被添加10次

   lazy loading:

   ```
   x = tf.Variable(10, name='x')
   y = tf.Variable(20, name='y')
   with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
   for _ in range(10):
   sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
   writer.close()
   ```

   add操作在图中仅被添加1次

   解决方法：

   1. 将计算操作和执行操作的定义分开
   2. 使用Python property来保证函数也只在第一次被调用的时候被load（不懂）

参考文献：

Stanford cs20si_2017 slides_02

https://web.stanford.edu/class/cs20si/2017/lectures/slides_02.pdf
