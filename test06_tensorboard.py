import tensorflow as tf

a = tf.Variable(0)
c = tf.assign_add(a, 1)
d = tf.summary.scalar('d', a)  # 收集标量

merge = tf.summary.merge_all()  # 自动管理

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)
    sess.run(init)
    for i in range(100):
        print(sess.run(c))
        summary = sess.run(merge)
        writer.add_summary(summary, i)
