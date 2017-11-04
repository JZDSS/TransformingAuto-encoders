import tensorflow as tf
import numpy as np

from TransformingAutoEncoder import TransformingAutoEncoder as TAE
from Trainer import Trainer

flags = tf.app.flags

flags.DEFINE_string('logdir', './logs', 'Log direction')
flags.DEFINE_string('ckptdir', './ckpt', 'Check point direction')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('total_steps', 100000, 'Number of steps to train')
flags.DEFINE_float('weight_decay', 0.0001, 'Weight decay')
flags.DEFINE_float('momentum', 0.95, 'Momentum')

FLAGS = flags.FLAGS

sess = tf.Session()


def main(_):

    encoder = TAE()
    encoder.forward()

    trainer = Trainer(encoder, FLAGS.learning_rate, FLAGS.weight_decay, FLAGS.momentum, FLAGS.batch_size,
                      FLAGS.total_steps, FLAGS.ckptdir, FLAGS.logdir)

    trainer.data_preparation()
    trainer.train()



if __name__ == "__main__":

    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
        tf.gfile.MakeDirs(FLAGS.logdir)

    tf.app.run()

