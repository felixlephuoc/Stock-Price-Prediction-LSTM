
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluate_ts import evaluate_ts
from tools import fetch_stock_price, format_dataset

tf.reset_default_graph() # Clear the default graph and resets the global default graph
tf.set_random_seed(101)

# Settings for the dataset creation
symbol = "AAPl"
feat_dimension = 20
train_size = 252
test_size = 252 - feat_dimension

# Fetch the values, and prepare the  train/test split
stock_values = fetch_stock_price(symbol, datetime.date(2015,1,1), datetime.date(2016,12,31))
minibatch_X, minibatch_y = format_dataset(stock_values, feat_dimension)

train_X = minibatch_X[:train_size, :].astype(np.float32)
train_y = minibatch_y[:train_size].reshape((-1,1)).astype(np.float32)
test_X = minibatch_X[train_size:, :].astype(np.float32)
test_y = minibatch_y[train_size:].reshape((-1,1)).astype(np.float32)

# Define the place holders
X_tf = tf.placeholder("float", shape=(None, feat_dimension), name="X")
y_tf = tf.placeholder("float", shape = (None, 1), name="y")

# Setting for the tensorflow
learning_rate = 0.1
optimizer = tf.train.AdamOptimizer
n_epochs = 10000

# define the simple linear regression model
def regression_ANN(x, weights, biases):
    return tf.add(biases, tf.matmul(x, weights))

# Store weights and bias
weights = tf.Variable(tf.truncated_normal([feat_dimension, 1], mean=0.0, stddev=1.0), name="weights")
biases = tf.Variable(tf.zeros([1,1]), name="bias")

# predictions, cost and operator
y_pred = regression_ANN(X_tf, weights, biases)
cost = tf.reduce_mean(tf.square(y_tf - y_pred))
train_op = optimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # For each epoch, the whole training set is feeded into the tensorflow graph
    for i in range(n_epochs):
        train_cost, _ = sess.run([cost, train_op], feed_dict={X_tf: train_X, y_tf: train_y})
        print("Training iteration", i, "MSE", train_cost)

    # After the training, check performance of the test set
    test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf: test_X, y_tf: test_y})
    print("Test dataset:", test_cost)

    # Evaluate the results
    evaluate_ts(test_X, test_y, y_pr)

    print("Shape of y_pr: ", y_pr.shape)
    # Visualize the predicted values
    plt.plot(range(len(stock_values)), stock_values, 'b')
    plt.plot(range(len(stock_values)-test_size, len(stock_values)), y_pr, 'r--')
    plt.xlabel("Days")
    plt.ylabel("Stock Price (USD)")
    plt.title("Predicted (Red) VS Real Blue on train & test set - Regression")
    plt.savefig("./graph/regression/prediction_on_train_test_set.png")
    plt.show()

