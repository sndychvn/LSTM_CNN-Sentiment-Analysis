import tensorflow as tf
import numpy as np

class CNN(object):
                                                                                                ##Used an embedding layer, a Convolutional Layer, max-pooling and Softmax layer
    def __init__(self.sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")        #Placeholders for input, output and dropout
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)                                                               #keeping track of L2 regularization loss (optional)

        #1. EMBEDDING LAYER
        with tf.device('/cpu:0'), tf.name_scope("embedding"):                                   #allocating GPUs and using embedding layers
            self.W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_size], minval=-1.0, maxval=1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)                  #explaination lookup https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)              #adding another dimension to the Tensor (expand_dims)
        pooled_outputs = []

        #2. CONVOLUTIONAL LAYER
        for i, filter_size in enumerate(filter_sizes):                                          #size of the filter (3x3 or 5x5 ......)
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")         #creating a 4D Tensor for weights
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")                 #creating a bias with 0.1 as value with a shape equal to number of filters (i.e if num_filters = 100) shape would be (100,1)

                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")        #providing a 4D Tensor to the CONV Layer

                h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")                               #apply non-linearity ReLu as pooling requires a non-linearity to be added

                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        num_filters_tool = num_filters * len(filter_sizes)                                        #combine all pooled feature
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_tool])


        with tf.name_scope("dropout"):                                                             #adding dropout
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        with tf.name_scope("output"):                                                               #Final (unnormalized) scores and predictions
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        with tf.name_scope("loss"):                                                                 # Calculate Mean cross-entropy loss
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):                                                             # Accuracy
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        print("CNN is Loaded!")