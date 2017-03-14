# %matplotlib inline
# import scipy.io as sp
import os
import numpy as np
import tensorflow as tf
# import cv2
from scipy import ndimage
from spatial_transformer import transformer
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import scipy.io as io
import matplotlib.pyplot as plt
from skimage.io import imread
from random import shuffle
from tensorflow.contrib.rnn import GRUCell

# Loading Data

trainset = io.loadmat("../Data/train_aim_qlty_sequence.mat")
testset = io.loadmat("../Data/test_aim_qlty_sequence.mat")

# Network Parameters

learning_rate = 0.01
n_epochs = 1000
batch_size = 100
display_step = 10

seq_max_len = 160 # Sequence max length
n_hidden = 40
max_length = 160
frame_size = 3
num_hidden = 40

def create_indexes(list_sequence_length, seq_max_len):
    indexes = []
    for i, sequence_length in enumerate(list_sequence_length):
        indexes += [sequence_length + seq_max_len*i - 1]
    return indexes

# Defining Placeholders

x_chemical = tf.placeholder(tf.float32, [None, 14])
x_process = tf.placeholder(tf.float32, [None, 14])
x_cooling = tf.placeholder(tf.float32, [None, max_length, frame_size])
y = tf.placeholder(tf.float32, [None, 3])
list_seqlen = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


# Defining Model

# RNN

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, 3]))
}
biases = {
    'out': tf.Variable(tf.random_normal([3]))
}


output, state = tf.nn.dynamic_rnn(
    GRUCell(num_hidden),
    sequence,
    dtype=tf.float32,
    sequence_length=length(sequence),
)

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = length
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

num_classes = 3

last = last_relevant(output, list_seqlen)
weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
rnn_output = tf.matmul(last, weight) + bias

# MLP 1

W_mlp1_fc1 = tf.Variable(tf.truncated_normal([14, 10], stddev=0.1))
b_mlp1_fc1 = tf.Variable(tf.constant(0.1, shape=[10]))
h_mlp1_fc1 = tf.nn.relu(tf.matmul(x_chemical, W_mlp1_fc1) + b_mlp1_fc1)
h_mlp1_fc1_drop = tf.nn.dropout(h_mlp1_fc1, keep_prob)

W_mlp1_fc2 = tf.Variable(tf.truncated_normal([10, 5], stddev=0.1))
b_mlp1_fc2 = tf.Variable(tf.constant(0.1, shape=[5]))
h_mlp1_fc2 = tf.nn.relu(tf.matmul(h_mlp1_fc1, W_mlp1_fc2) + b_mlp1_fc2)
h_mlp1_fc2_drop = tf.nn.dropout(h_mlp1_fc2, keep_prob)


W_mlp1_fc3 = tf.Variable(tf.truncated_normal([5, 3], stddev=0.1))
b_mlp1_fc3 = tf.Variable(tf.constant(0.1, shape=[3]))
mlp1_output = tf.matmul(h_mlp1_fc2_drop, W_mlp1_fc3) + b_mlp1_fc3


# MLP 2

W_mlp2_fc1 = tf.Variable(tf.truncated_normal([14, 10], stddev=0.1))
b_mlp2_fc1 = tf.Variable(tf.constant(0.1, shape=[10]))
h_mlp2_fc1 = tf.nn.relu(tf.matmul(x_process, W_mlp2_fc1) + b_mlp2_fc1)
h_mlp2_fc1_drop = tf.nn.dropout(h_mlp2_fc1, keep_prob)


W_mlp2_fc2 = tf.Variable(tf.truncated_normal([10, 5], stddev=0.1))
b_mlp2_fc2 = tf.Variable(tf.constant(0.1, shape=[5]))
h_mlp2_fc2 = tf.nn.relu(tf.matmul(h_mlp2_fc1, W_mlp2_fc2) + b_mlp2_fc2)
h_mlp2_fc2_drop = tf.nn.dropout(h_mlp2_fc2, keep_prob)


W_mlp2_fc3 = tf.Variable(tf.truncated_normal([5, 3], stddev=0.1))
b_mlp2_fc3 = tf.Variable(tf.constant(0.1, shape=[3]))
mlp2_output = tf.matmul(h_mlp2_fc2, W_mlp2_fc3) + b_mlp2_fc3

# ensemble

concat_output = tf.concat([rnn_output, mlp1_output, mlp2_output], 1)
W_ensemble = tf.Variable(tf.truncated_normal([9, 3], stddev=0.1))
b_ensemble = tf.Variable(tf.constant(0.1, shape=[3]))
output = tf.matmul(concat_output, W_ensemble) + b_ensemble

loss = tf.to_double(tf.losses.mean_squared_error(output, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_error = []
    for epoch in range(n_epochs):
#         print epoch
        for i in range(len(trainset["chemical_parameters"])/batch_size):
#             print "   " + str(i)
            batch_chemical = trainset["chemical_parameters"][i*batch_size:(i+1)*batch_size]
            batch_process = trainset["process_parameters"][i*batch_size:(i+1)*batch_size]
            batch_cooling = trainset["cooling_sequence"][i*batch_size:(i+1)*batch_size]
            batch_y = trainset["output_parameters"][i*batch_size:(i+1)*batch_size]
            batch_seqlen = trainset["sequence_lengths"][0][i*batch_size:(i+1)*batch_size]
            sess.run(optimizer, feed_dict={x_chemical: batch_chemical, x_process: batch_process, x_cooling: batch_cooling, y: batch_y, list_seqlen: batch_seqlen, keep_prob: 0.5})
        train_cost = sess.run(loss, feed_dict={x_chemical: trainset["chemical_parameters"], x_process: trainset["process_parameters"], x_cooling: testset["cooling_sequence"], y: trainset["output_parameters"], list_seqlen: trainset["sequence_lengths"][0], keep_prob: 1.0})
        test_cost = sess.run(loss, feed_dict={x_chemical: testset["chemical_parameters"], x_process: testset["process_parameters"], x_cooling: testset["cooling_sequence"], y: testset["output_parameters"], list_seqlen: testset["sequence_lengths"][0], keep_prob: 1.0})
        print "Epoch:", '%04d' % (epoch + 1), "train error=", \
                "{:.9f}".format(train_cost), "test error=", \
                "{:.9f}".format(test_cost)
        train_error += [train_cost]
        test_error += [test_cost]
        

error_dict = {}
error_dict["train"] = train_error
error_dict["test"] = test_error

io.savemat("error_ensemble_tensorflow.mat", error)

