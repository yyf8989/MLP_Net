import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np


class MLPnet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, 100],stddev=tf.sqrt(1/100)), dtype=tf.float32)
        self.b1 = tf.zeros([100])

        self.w2 = tf.Variable(tf.truncated_normal(shape=[100, 40],stddev=tf.sqrt(4/40)),dtype=tf.float32)
        self.b2 = tf.zeros([40])

        self.out_w = tf.Variable(tf.truncated_normal(shape=[40, 10],stddev=tf.sqrt(4/10)), dtype=tf.float32)
        self.out_b = tf.zeros([10])

    def forward(self):
        self.fc1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)
        self.fc2 = tf.nn.relu(tf.matmul(self.fc1, self.w2) + self.b2)
        self.out_put = tf.nn.softmax(tf.matmul(self.fc2, self.out_w) + self.out_b)

    def backward(self):
        self.loss = tf.reduce_mean((self.out_put - self.y) ** 2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    net = MLPnet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(10001):
            train_x,  train_y = mnist.train.next_batch(100)
            _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x:train_x, net.y:train_y})

            if epoch % 100 == 0:
                test_x, test_y = mnist.test.next_batch(1000)
                test_out_y = sess.run(net.out_put, feed_dict={net.x:test_x})

                test_y = np.argmax(test_y, axis=1)
                test_out_y = np.argmax(test_out_y, axis=1)

                accuracy = np.mean(np.array(test_y == test_out_y), dtype=np.float32)
                print("accuracy:", accuracy)