import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

import os

import TransformingAutoEncoder


class Trainer():
    def __init__(self, model, learning_rate, weight_decay, momentum, batch_size, total_steps, ckptdir, logdir):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.total_steps = total_steps
        self.ckptdir = ckptdir
        self.batch_size = batch_size
        self.logdir = logdir
        self.train_writer = None
        self.test_writer = None
        self.input = None
        self.expectation = None
        self.shift = None
        self.data_preparation()

    def data_preparation(self):
        mnist = input_data.read_data_sets('data', one_hot=True,
                                          source_url='http://yann.lecun.com/exdb/mnist/')
        def translate(input, shift=5, legth=28):
            X = np.reshape(input, (len(input), legth, legth))
            expectation, shift_list, original = [], [], []

            for i in np.random.permutation(len(X)):
                trans_x = np.random.randint(-shift, shift)
                trans_y = np.random.randint(-shift, shift)

                trans_img = np.roll(np.roll(X[i], trans_x, axis=0), trans_y, axis=1)
                expectation.append((trans_img.flatten()-0.5)*2)
                original.append((input[i] - 0.5) * 2)
                shift_list.append((trans_x, trans_y))

            return np.array(expectation), np.array(shift_list), np.array(original)

        self.expectation, self.shift, self.input = translate(mnist.train.images)
        self.expectation_v, self.shift_v, self.input_v = translate(mnist.validation.images)

    def get_batch(self, train=True):
        if train:
            id = np.random.randint(low=0, high=self.input.shape[0], size=self.batch_size, dtype=np.int32)
            return {self.model.input: self.input[id, ...],
                    self.model.shift: self.shift[id, ...],
                    self.model.expectation: self.expectation[id, ...]}
        else:
            id = np.random.randint(low=0, high=self.input_v.shape[0], size=self.batch_size, dtype=np.int32)
            return {self.model.input: self.input_v[id, ...],
                    self.model.shift: self.shift_v[id, ...],
                    self.model.expectation: self.expectation_v[id, ...]}

    def train(self):
        with tf.name_scope('train'):
            global_step = tf.Variable(1, name="global_step")
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       global_step, 500, 0.98, True, "learning_rate")
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.model.loss, global_step=global_step)
            tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            self.train_writer = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
            self.test_writer = tf.summary.FileWriter(self.logdir + '/test', sess.graph)

            self.train_writer.flush()
            self.test_writer.flush()

            saver = tf.train.Saver(name="saver")
            if tf.gfile.Exists(os.path.join(self.ckptdir, 'checkpoint')):
                saver.restore(sess, os.path.join(self.ckptdir, 'model.ckpt'))
            else:
                tf.gfile.MkDir(self.ckptdir)
                sess.run(tf.global_variables_initializer())

            for i in range(self.total_steps):
                sess.run(train_step, feed_dict=self.get_batch())
                if i % 1000 == 0 and i != 0:  # Record summaries and test-set accuracy
                    summary = sess.run(merged, feed_dict=self.get_batch())
                    self.train_writer.add_summary(summary, i)

                    summary = sess.run(merged, feed_dict=self.get_batch(False))
                    self.test_writer.add_summary(summary, i)

                    saver.save(sess, os.path.join(self.ckptdir, 'model.ckpt'))

