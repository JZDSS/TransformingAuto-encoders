import tensorflow as tf
import tensorflow.contrib.layers as layers
import math


class Capsule:
    def __init__(self, dim_input, num_recognise_units, num_generation_units, activation_fn, weight_decay, name):

        self.dim_input = dim_input
        self.num_recognise_units = num_recognise_units
        self.num_generation_units = num_generation_units
        self.activation_fn = activation_fn
        self.weight_decay = weight_decay
        self.name = name
        self.output = None

    def forward(self, input, shift):
        with tf.variable_scope(self.name):

            # recognition
            recognition = layers.fully_connected(inputs=input, num_outputs=self.num_recognise_units,
                                                 activation_fn=self.activation_fn, scope='recognition',
                                                 weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=0.001),
                                                 biases_initializer=tf.truncated_normal_initializer(
                                                     stddev=0.001),
                                                 weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                 biases_regularizer=layers.l2_regularizer(self.weight_decay))

            # predict location
            location = layers.fully_connected(inputs=recognition, num_outputs=2,
                                                activation_fn=self.activation_fn, scope='location',
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=0.001),
                                                biases_initializer=tf.truncated_normal_initializer(
                                                    stddev=0.001),
                                                weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                biases_regularizer=layers.l2_regularizer(self.weight_decay))
            location = location + tf.cast(shift, tf.float32)
            # predict probability
            probability = layers.fully_connected(recognition, num_outputs=1,
                                                activation_fn=self.activation_fn, scope='probability',
                                                weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=0.001),
                                                biases_initializer=tf.truncated_normal_initializer(
                                                     stddev=0.001),
                                                weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                biases_regularizer=layers.l2_regularizer(self.weight_decay))
            probability = tf.tile(probability, [1, self.dim_input])

            # generation
            generation = layers.fully_connected(inputs=location, num_outputs=self.num_generation_units,
                                                activation_fn=self.activation_fn, scope='generation',
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=0.001),
                                                biases_initializer=tf.truncated_normal_initializer(
                                                    stddev=0.001),
                                                weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                biases_regularizer=layers.l2_regularizer(self.weight_decay))

            # output
            output = layers.fully_connected(inputs=generation, num_outputs=self.dim_input,
                                                activation_fn=self.activation_fn, scope='output',
                                                weights_initializer=tf.truncated_normal_initializer(
                                                     stddev=0.001),
                                                biases_initializer=tf.truncated_normal_initializer(
                                                     stddev=0.001),
                                                weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                                biases_regularizer=layers.l2_regularizer(self.weight_decay))

            self.output = tf.multiply(output, probability)

            return self.output
