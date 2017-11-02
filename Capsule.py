import tensorflow as tf
import tensorflow.contrib.layers as layers
import math


class Capsule:
    def __init__(self, dim_input, num_recognise_units, num_generation_units, activation, weight_decay, name):

        self.input = tf.placeholder(tf.float32, shape=[None, 784], name='input')
        self.shift = tf.placeholder(tf.float32, shape=[None, 2], name='shift')
        self.expectation = tf.placeholder(tf.float32, shape=[None, 784], name='expectation')
        self.dim_input = dim_input
        self.num_recognise_units = num_recognise_units
        self.num_generation_units = num_generation_units
        self.activation = activation
        self.weight_decay = weight_decay
        self.name = name

    def build(self):
        with tf.variable_scope(self.name):

            # recognition
            recognition = layers.fully_connected(inputs=self.input, num_outputs=self.num_recognise_units,
                                                 activation_fn=tf.sigmoid, scope='recognition',
                                                 weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=math.sqrt(2. / self.dim_input / self.num_recognise_units)),
                                                 weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                 biases_regularizer=layers.l2_regularizer(self.weight_decay))

            # predict location
            location = layers.fully_connected(inputs=recognition, num_outputs=2,
                                                 activation_fn=tf.sigmoid, scope='location',
                                                 weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=math.sqrt(2. / self.num_recognise_units / 2.)),
                                                 weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                 biases_regularizer=layers.l2_regularizer(self.weight_decay))

            # predict probability
            probability = layers.fully_connected(recognition, num_outputs=1,
                                                 activation_fn=tf.sigmoid, scope='probability',
                                                 weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=math.sqrt(2. / self.num_recognise_units / 1)),
                                                 weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                 biases_regularizer=layers.l2_regularizer(self.weight_decay))
            probability = tf.tile(probability, [1, self.dim_input])

            # generation
            generation = layers.fully_connected(inputs=location, num_outputs=self.num_generation_units,
                                                 activation_fn=tf.sigmoid, scope='generation',
                                                 weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=math.sqrt(2. / 2 / self.num_generation_units)),
                                                 weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                 biases_regularizer=layers.l2_regularizer(self.weight_decay))

            # output
            output = layers.fully_connected(inputs=generation, num_outputs=self.dim_input,
                                                activation_fn=tf.sigmoid, scope='output',
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=math.sqrt(2. / self.num_generation_units / self.dim_input)),
                                                weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                biases_regularizer=layers.l2_regularizer(self.weight_decay))

            self.output = tf.multiply(output, probability)
