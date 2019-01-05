import tensorflow as tf
import numpy as np

class CNN(object):              #Used an embedding layer, a Convolutional Layer, max-pooling and Softmax layer

    def __init__(self,sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")        #Placeholders for input, output and dropout
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)                                                              #keeping track of L2 regularization loss (optional)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):                                   #allocating GPUs and using embedding layers
            self.W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1.0, maxval=1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            





