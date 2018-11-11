# 分类空间
# import tensorflow as tf

# a = tf.get_variable("a",shape=[2])
# b = tf.get_variable("b",shape=[2])
# c = tf.get_variable("c",shape=[2])
#
# tf.add_to_collection("group_1",a)
# tf.add_to_collection("group_1",b)
# tf.add_to_collection("group_2",c)
#
# print(tf.get_collection("group_1"))
# print(tf.get_collection("group_2"))


import tensorflow as tf

a = tf.get_variable('a', dtype=tf.float32, shape=[2])
b = tf.get_variable('b', dtype=tf.float32, shape=[2])
c = tf.get_variable('c', dtype=tf.float32, shape=[2])

tf.add_to_collection('group1', a)
tf.add_to_collection('group1', b)
tf.add_to_collection('group2', c)

print(tf.get_collection('group1'))
print(tf.get_collection('group2'))
