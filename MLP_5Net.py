import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  tensorflow as tf
import  tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import numpy as np

class MLPnet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, 100], stddev=0.1), dtype=tf.float32)
        self.b1 = tf.Variable(np.zeros([100]), dtype=tf.float32)

        self.w2 = tf.Variable(tf.truncated_normal(shape=[100, 60], stddev=0.1), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros([60]), dtype=tf.float32)

        self.w3 = tf.Variable(tf.truncated_normal(shape=[60, 40], stddev=0.1), dtype=tf.float32)
        self.b3 = tf.Variable(np.zeros([40]), dtype=tf.float32)

        self.w4 = tf.Variable(tf.truncated_normal(shape=[40, 30], stddev=0.1), dtype=tf.float32)
        self.b4 = tf.Variable(np.zeros([30]), dtype=tf.float32)

        self.out_w = tf.Variable(tf.truncated_normal(shape=[30,10], stddev=0.1), dtype=tf.float32)
        self.out_b = tf.Variable(np.zeros([10]), dtype=tf.float32)

    def forward(self):
        self.fc1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)
        self.fc2 = tf.nn.relu(tf.matmul(self.fc1, self.w2) + self.b2)
        self.fc3 = tf.nn.relu(tf.matmul(self.fc2, self.w3) + self.b3)
        self.fc4 = tf.nn.relu(tf.matmul(self.fc3, self.w4) + self.b4)
        self.output = tf.nn.softmax(tf.matmul(self.fc4, self.out_w) + self.out_b)

    def backward(self):
        self.loss = tf.reduce_mean((self.output - self.y)**2)
        self.opt = tf.train.GradientDescentOptimizer(0.2).minimize(self.loss)


if __name__ == '__main__':
    net = MLPnet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(100001):
            xs, ys = mnist.train.next_batch(1000)
            _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x:xs, net.y:ys})

            if epoch % 1000 == 0:
                test_xs, test_ys = mnist.test.next_batch(1000)
                test_out = sess.run(net.output, feed_dict={net.x: test_xs})

                test_y = np.argmax(test_ys, axis=1)
                test_out = np.argmax(test_out, axis=1)

                accuracy = np.mean(np.array(test_y == test_out, dtype=np.float32))
                print("epoch:{}".format(epoch))
                print("accuracy:{}".format(accuracy))




