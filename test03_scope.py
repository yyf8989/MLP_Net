import tensorflow as tf

with tf.variable_scope('abc'):
    a = tf.get_variable('cc', shape=[2])
    b = tf.get_variable('ac', shape=[2])

with tf.variable_scope('abc', reuse=True):
    c = tf.get_variable('cc', shape=[2])

print(a)
print(b)
print(c)

op = tf.assign(a, [1, 2])
op1 = tf.assign(b, [2, 20])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(op)
    sess.run(op1)
    print(sess.run(b))
    print(sess.run(a))
