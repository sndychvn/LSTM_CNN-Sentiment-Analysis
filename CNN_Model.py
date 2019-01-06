import tensorflow as tf
import numpy as np

class CNN(object):
                                                                                                ##Used an embedding layer, a Convolutional Layer, max-pooling and Softmax layer
    def __init__(self.sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")        #Placeholders for input, output and dropout
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)                                                              #keeping track of L2 regularization loss (optional)
        with tf.device('/cpu:0'), tf.name_scope("embedding"):                                   #allocating GPUs and using embedding layers
            self.W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1.0, maxval=1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)                  #explaination lookup https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)              #adding another dimension to the Tensor (expand_dims)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")         #creating a 4D Tensor
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")                               #apply nonlinearity
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        num_filters_tool = num_filters * len(filter_sizes)                                        #combine all pooled feature
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_tool])










