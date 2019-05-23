

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluate_ts import evaluate_ts
from tensorflow.contrib import rnn
from tools import fetch_stock_price, format_dataset

tf.reset_default_graph()
tf.set_random_seed(101)
np.random.seed(101)

# Setting for the dataset creation
symbol = "AAPL"
time_dimension = 20
train_size = 504
test_size = 252 - time_dimension

# Setting for tensorflowjp
tf_logdir = "./logs/tf/stock_price_lstm"
os.makedirs(tf_logdir, exist_ok=1)
learning_rate = 0.05
optimizer = tf.train.AdagradOptimizer
n_epochs = 5000
n_embeddings = 128

# Fetch the values, and prepare the train/test split
stock_values = fetch_stock_price(symbol, datetime.date(2014,1,1), datetime.date(2016,12,31))
minibatch_cos_X, minibatch_cos_y = format_dataset(stock_values, time_dimension)

train_X = minibatch_cos_X[:train_size,:].astype(np.float32)
train_y = minibatch_cos_y[:train_size].reshape((-1,1)).astype(np.float32)
test_X = minibatch_cos_X[train_size:,:].astype(np.float32)
test_y = minibatch_cos_y[train_size:].reshape((-1,1)).astype(np.float32)

train_X_ts = train_X[:,:,np.newaxis]
test_X_ts = test_X[:,:,np.newaxis]

# Create the placeholders
X_tf = tf.placeholder("float", shape=(None, time_dimension, 1), name="X")
y_tf = tf.placeholder("float", shape=(None, 1), name="y")

# The LSTM model
def RNN(x, weights, biases):
    with tf.name_scope("LSTM"):
        x_ = tf.unstack(x, time_dimension, 1)
        lstm_cell = rnn.BasicLSTMCell(n_embeddings)
        outputs, _ = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)
        return tf.add(biases, tf.matmul(outputs[-1], weights))

# Store layers weights and biases
weights = tf.Variable(tf.truncated_normal([n_embeddings, 1], mean=0.0, stddev=10.0), name="weights")
biases = tf.Variable(tf.zeros([1]), name="bias")

# Model, cost and optimizer
y_pred = RNN(X_tf, weights, biases)
# Configuration for tensorboard
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.square(y_tf - y_pred))
    train_op = optimizer(learning_rate).minimize(cost)
    tf.summary.scalar("MSE", cost)

with tf.name_scope("mae"):
    mae_cost = tf.reduce_mean(tf.abs(y_tf - y_pred))
    tf.summary.scalar("mae", mae_cost)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(tf_logdir, sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    # For each epoch, the whole training set is feeded into the tensorflow graph
    for i in range(n_epochs):
        summary, train_cost, _ = sess.run([merged, cost, train_op], feed_dict={X_tf: train_X_ts, y_tf: train_y})
        writer.add_summary(summary, i)
        if i%100 == 0:
            print("Training iteration", i, "MSE", train_cost)

    # After training, check the performance on the test set
    test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf:test_X_ts, y_tf:test_y})
    print("Test dataset:", test_cost)

    # Evaluate the result
    evaluate_ts(test_X, test_y, y_pr)

    # Visualize the predicted value on both training and test set
    plt.plot(range(len(stock_values)), stock_values, 'b')
    plt.plot(range(len(stock_values) - test_size, len(stock_values)), y_pr, 'r--')
    plt.xlabel("Days")
    plt.ylabel("Stock Price (USD)")
    plt.title("Predicted (Red) VS Real Blue on train & test set - LSTM")
    plt.savefig("./graph/rnn/predicted_vs_real_values_train_test_set.png")
    plt.show()


# Launch the tensorboard by typing the following command in the terminal:
# tensorboard --logdir=./logs/tf/stock_price_lstm
# open the browser to http://127/0.0.1:6006
