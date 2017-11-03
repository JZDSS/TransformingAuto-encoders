import tensorflow as tf
import numpy as np

from TransformingAutoEncoder import TransformingAutoEncoder as tae

flags = tf.app.flags

flags.DEFINE_string('logdir', './logs', 'Log direction')
flags.DEFINE_string('ckpt', './ckpt', 'Check point direction')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('batch', 100, 'Batch size')
flags.DEFINE_integer('total_steps', 10000, 'Number of steps to train')

FLAGS = flags.FLAGS

sess = tf.Session()


def main(_):

    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
        tf.gfile.MakeDirs(FLAGS.logdir)

    encoder = tae()
    encoder.forward()

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(FLAGS.logdir)
    merged = tf.summary.merge_all()
    writer.add_graph(sess.graph)
    writer.add_summary(sess.run(merged, feed_dict={encoder.input: np.random.uniform(-1, 1, [40, 784]),
                                                   encoder.shift: np.random.randint(0, 3, [40, 2]),
                                                   encoder.expectation: np.random.uniform(-1, 1, [40, 784])}), 0)
    writer.close()
    # writer.flush()

if __name__ == "__main__":
    tf.app.run()

