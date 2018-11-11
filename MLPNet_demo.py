import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np


class MLPNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        with tf.name_scope("fc1"):
            self.in_w = tf.Variable(tf.truncated_normal(shape=[784, 100], stddev=tf.sqrt(1 / 100)))
            self.in_b = tf.Variable(tf.zeros([100]))

            self.summary("in_w", self.in_w)

        with tf.name_scope("fc2"):
            self.out_w = tf.Variable(tf.truncated_normal(shape=[100, 10], stddev=tf.sqrt(1 / 10)))
            self.out_b = tf.Variable(tf.zeros([10]))

            self.summary("out_w", self.out_w)

    def forward(self):
        with tf.name_scope("fc1"):
            self.fc1 = tf.nn.relu(tf.matmul(self.x, self.in_w) + self.in_b)

        with tf.name_scope("fc2"):
            self.output = tf.nn.softmax(tf.matmul(self.fc1, self.out_w) + self.out_b)

    def backward(self):
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean((self.output - self.y) ** 2)
            tf.summary.scalar("loss", self.loss)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def summary(self, name, w):
        tf.summary.histogram(name + "_w", w)
        tf.summary.scalar(name + "_max", tf.reduce_max(w))
        tf.summary.scalar(name + "_min", tf.reduce_min(w))
        tf.summary.scalar(name + "_mean", tf.reduce_mean(w))


if __name__ == '__main__':
    net = MLPNet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./logs", sess.graph)
        for epoch in range(1000):
            xs, ys = mnist.train.next_batch(200)
            # test_output = sess.run(net.output,feed_dict={net.x:xs,net.y:ys})
            summary, _loss, _ = sess.run([merged, net.loss, net.opt], feed_dict={net.x: xs, net.y: ys})
            writer.add_summary(summary, epoch)

            if epoch % 100 == 0:
                test_xs, test_ys = mnist.test.next_batch(200)
                test_output = sess.run(net.output, feed_dict={net.x: test_xs})
                test_y = np.argmax(test_ys, axis=1)
                test_out = np.argmax(test_output, axis=1)
                test_accuracy = np.mean(np.array(test_y == test_out, dtype=np.float32))
                tf.summary.scalar('test_accuracy', test_accuracy)
                # print("标签",test_y,"结果",test_out)
