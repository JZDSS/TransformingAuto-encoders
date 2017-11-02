import tensorflow as tf

from Capsule import Capsule

flags = tf.app.flags

flags.DEFINE_string('logdir', './logs', 'Log direction')
flags.DEFINE_string('ckpt', './ckpt', 'Check point direction')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('batch', 100, 'Batch size')
flags.DEFINE_integer('epoch', 100, 'Number of epochs to train')

FLAGS = flags.FLAGS

sess = tf.InteractiveSession()


def main(_):

    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
        tf.gfile.MakeDirs(FLAGS.logdir)

    capsule = Capsule(784, 10, 20, tf.sigmoid, 0.95, 'capsule1')
    capsule.build()

    writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    writer.flush()

if __name__ == "__main__":
    tf.app.run()