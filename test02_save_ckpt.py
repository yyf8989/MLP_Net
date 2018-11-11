#保存与恢复
import tensorflow as tf
import numpy as np

# w = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32,name='ABC')
# b = tf.Variable([[2,3,4]],dtype=tf.float32,name='BCE')
# init = tf.global_variables_initializer()
#
# save = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = save.save(sess,'./my_net/save_net.ckpt')
#     print('保存路径：',save_path)


w1 = tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32,name='ABC')
b1 = tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32,name='BCE')

#不需要对变量进行初始化
save = tf.train.Saver()
with tf.Session() as sess:
    save.restore(sess,'./my_net/save_net.ckpt')
    print('weight:',sess.run(w1))
    print('bias:', sess.run(b1))