import tensorflow as tf
import math

from Capsule import Capsule


class TransformingAutoEncoder:

    def __init__(self, dim_input=784, num_recognise_units=10, num_generation_units=20, activation_fn=tf.tanh,
                 name='TransformingAutoEncoder', num_capsules=60, weight_decay=0.95):

        self.dim_input = dim_input
        self.input = tf.placeholder(tf.float32, shape=[None, self.dim_input], name='input')
        self.shift = tf.placeholder(tf.int32, shape=[None, 2], name='shift')
        self.expectation = tf.placeholder(tf.float32, shape=[None, self.dim_input], name='expectation')
        self.num_capsules = num_capsules
        self.capsules = []
        self.capsule_outputs = []
        self.num_recognise_units = num_recognise_units
        self.num_generation_units = num_generation_units
        self.activation_fn = activation_fn
        self.name = name
        self.prediction = None
        self.loss = None
        self.weight_decay = weight_decay

    def forward(self):
        tf.summary.image('origin', tf.reshape(self.input,
                                              [-1, int(math.sqrt(self.dim_input)),
                                               int(math.sqrt(self.dim_input)), 1]), 1)
        with tf.variable_scope(self.name):
            for i in range(self.num_capsules):
                capsule = Capsule(self.dim_input, self.num_recognise_units, self.num_generation_units,
                                  self.activation_fn, self.weight_decay, ('capsule_%02d' % i))
                capsule_output = capsule.forward(self.input, self.shift)
                self.capsules.append(capsule)
                self.capsule_outputs.append(capsule_output)

            with tf.variable_scope('loss'):
                self.prediction = self.activation_fn(tf.add_n(self.capsule_outputs) / self.num_capsules)
                tf.losses.mean_squared_error(self.expectation, self.prediction)
                self.loss = tf.losses.get_total_loss()

                tf.summary.scalar('loss', self.loss)
                tf.summary.image('expectation', tf.reshape(self.expectation,
                                                           [-1, int(math.sqrt(self.dim_input)),
                                                            int(math.sqrt(self.dim_input)), 1]), 1)
                tf.summary.image('prediction', tf.reshape(self.prediction,
                                                          [-1, int(math.sqrt(self.dim_input)),
                                                           int(math.sqrt(self.dim_input)), 1]), 1)


