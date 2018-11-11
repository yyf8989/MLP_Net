# 共享变量
import tensorflow as tf

a = tf.Variable([1, 2, 3], name='a')
b = tf.Variable([2, 3, 4], name='a')
c = tf.Variable([4, 5, 6], name='a')
print(a)
print(b)
print(c)

a1 = tf.get_variable('bce', [12, 3, 4])
b1 = tf.get_variable('bcf', [23, 45, 6])
print(a1)
print(b1)
# 这两个是不能共享变量的，如果name一样，则会出错
