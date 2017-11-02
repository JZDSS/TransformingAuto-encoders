import tensorflow as tf

from Capsule import Capsule


class TransformingAutoencoder:

    def __init__(self, dim_input=784, num_recognise_units=10, num_generation_units=20, activation_fn=tf.tanh,
                 weight_decay=0.95, name='TransformingAutoencoder', num_capsules=60, learning_rate=0.001):

        self.dim_input = dim_input
        self.input = tf.placeholder(tf.float32, shape=[None, self.dim_input], name='input')
        self.shift = tf.placeholder(tf.float32, shape=[None, 2], name='shift')
        self.expectation = tf.placeholder(tf.float32, shape=[None, self.dim_input], name='expectation')

        self.num_capsules = num_capsules
        self.capsules = []
        self.num_recognise_units = num_recognise_units
        self.num_generation_units = num_generation_units
        self.activation_fn = activation_fn
        self.weight_decay = weight_decay
        self.name = name
        self.learning_rate = learning_rate
        self.prediction = None

    def forward(self):
        with tf.variable_scope(self.name):
            for i in range(self.num_capsules):
                capsule = Capsule(self.dim_input, self.num_recognise_units, self.num_generation_units,
                                  self.activation_fn, self.weight_decay, ('capsule_%02d' % i))
                capsule.forward(self.input, self.shift)
                self.capsules.append(capsule)

        self.prediction = self.activation_fn(tf.add_n(self.capsules))
