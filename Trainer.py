import tensorflow as tf

import TransformingAutoEncoder


class Trainer():
    def __init__(self, loss, learning_rate, weight_decay, momentum):
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
