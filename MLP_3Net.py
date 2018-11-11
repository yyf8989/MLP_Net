import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np


class MLPnet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.w1 = tf.Variable(tf.truncated_normal(dtype=tf.float32,shape=[784, 100], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([100]))

        self.w2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[100, 60], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([60]))

        self.out_w = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[60, 10], stddev=0.1))
        self.out_b = tf.Variable(tf.zeros([10]))

    def forward(self):
        self.fc1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)
        self.fc2 = tf.nn.relu(tf.matmul(self.fc1, self.w2) + self.b2)
        self.output = tf.nn.softmax(tf.matmul(self.fc2, self.out_w) + self.out_b)

    def backward(self):
        self.loss = tf.reduce_mean((self.output-self.y)**2)
        self.opt = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)


if __name__ == '__main__':
    net = MLPnet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(100001):
            train_x, train_y = mnist.train.next_batch(1000)
            _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x: train_x, net.y: train_y})

            if epoch % 1000 == 0:
                test_x, test_y = mnist.test.next_batch(10000)
                test_output = sess.run(net.output, feed_dict={net.x: test_x})

                test_y_labels = np.argmax(test_y, axis=1)
                test_out = np.argmax(test_output, axis=1)

                accuracy = np.mean(np.array(test_out == test_y_labels,dtype=np.float32))
                print('epoch:{}'.format(epoch))
                print('accuracy:{}'.format(accuracy))

