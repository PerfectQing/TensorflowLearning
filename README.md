# TensorflowLearning


## 第二章

### tf.where()

条件语句真返回A,条件语句假返回B

tf.where(条件语句, 真返回A, 假返回B)
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = <span style='color: yellow;'>tf.where</span>(tf.greater(a, b), a, b)  # 若 a > b, 返回 a 对应位置的元素,否则返回 b 对应的元素\

```python
运行结果:

```



### np.random.RandomState.rand()
返回一个 [0, 1) 之间的随机数
np.random.RandomState.rand(维度)  # 维度为空，返回标量

```python
import numpy as np
rdm = np.random.RandomState(seed=1)  # seed 定义生成随机数
a = rdm.rand()  # 返回一个标量
b = rdm.rand(2, 3) # 返回维度为2行3列随机数矩阵
print('a:', a)
print('b:', b)
```
```
运行结果:
a: 0.417022004702574
b: [[7.20324493e-01 1.14374817e-04 3.02332573e-01]
	[1.46755891e-01 9.23385948e-02 1.86260211e-01]]
```

### np.vstack()

将两个数组按垂直方向叠加

np.vstack(数组1， 数组2)

```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print('c:', c)
```

```
运行结果:
c: [[1 2 3]
 	[4 5 6]]
```

### np.mgrid[]   .ravle()    np.c_[]

np.mgrid[]

np.mgrid[起始值: 结束值: 步长, 起始值: 结束值: 步长, ...]

.ravle()   将x变为一维数组，”把 .前变量拉直“

np.c_[]  使返回的间隔数值点配对

np.c_[数组1, 数组2, ...]

```python
import numpy as np
x, y = np.mgrid[1: 3: 1, 2: 4: 0.5]
grid = np.c_[x.ravel(), y.ravel()]
print('x:', x)
print('y:', y)
print('grid:\n', grid)
```

```
运行结果:
x: [[1. 1. 1. 1.]
	[2. 2. 2. 2.]]
y: [[2.  2.5 3.  3.5]
	[2.  2.5 3.  3.5]]
grid:
 [[1.  2. ]
  [1.  2.5]
  [1.  3. ]
  [1.  3.5]
  [2.  2. ]
  [2.  2.5]
  [2.  3. ]
  [2.  3.5]]
```

