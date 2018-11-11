import tensorflow as tf

with tf.variable_scope('xyz', dtype=tf.float32) as scop:
    a = tf.get_variable('a', shape=[2], trainable=False)
    b = tf.get_variable('b', shape=[2])

c = tf.get_variable('c', shape=[2])

print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=('xyz')))
