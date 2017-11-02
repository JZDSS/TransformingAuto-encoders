import tensorflow as tf

class TransformingAutoendoder:
    def __init__(self, input, num_capsules,):
        self.num_capsules = num_capsules
        self.shift = tf.placeholder(tf.float32, shape=[None, 2], name='shift')
        self.expectation = tf.placeholder(tf.float32, shape=[None, 784], name='expectation')

    def forward(self):
        pass
