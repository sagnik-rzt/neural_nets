import tensorflow as tf
import pandas as pd

def get_data(n_examples = 300, train_fraction = 0.66):

    dataset = pd.read_csv(filepath_or_buffer = "mnist_train.csv")
    reduced_dataset = dataset.sample(n = n_examples, replace = False)

    x = reduced_dataset.iloc[:, 1 : 785]
    y = pd.get_dummies(data = reduced_dataset.iloc[:, 0])

    x_train = x.iloc[0 : int(n_examples * train_fraction), :]
    x_test = x.iloc[int(n_examples * train_fraction) + 1 : , :]

    y_train = y.iloc[0 : int(n_examples * train_fraction), :]
    y_test = y.iloc[int(n_examples * train_fraction) + 1 : , :]

    return x_train, x_test, y_train, y_test


def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape = shape))


def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape = shape))


def convolution2D(X, W):
    return tf.nn.conv2d(input = X, filter = W, strides = [1,1,1,1], padding = 'SAME')


def max_pool(X):
    return tf.nn.max_pool(value = X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')




X_data = tf.placeholder(tf.float32, shape = [None, 784])
Y_data = tf.placeholder(tf.float32, shape = [None, 10])
X_image = tf.reshape(tensor = X_data, shape = [-1, 28, 28, 1])

conv1_weights = weight_variable([5,5,1,32])
conv1_bias = bias_variable([32])
conv1_output = max_pool(tf.nn.relu(convolution2D(X_image, conv1_weights) + conv1_bias))

conv2_weights = weight_variable([5,5,32,64])
conv2_bias = bias_variable([64])
conv2_output = max_pool(tf.nn.relu(convolution2D(conv1_output, conv2_weights) + conv2_bias))

fc1_weights = weight_variable([7 * 7 * 64, 1024])
fc1_bias = bias_variable([1024])
conv2_output_flat = tf.reshape(tensor = conv2_output, shape = [-1, 7 * 7 * 64])
fc1_output = tf.nn.relu(tf.matmul(conv2_output_flat, fc1_weights) + fc1_bias)

fc2_weights = weight_variable([1024, 10])
fc2_bias = bias_variable([10])
Y_hat = tf.matmul(fc1_output, fc2_weights) + fc2_bias

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_data, logits = Y_hat))
update_weights = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)



with tf.Session() as sess:

    x_train, x_test, y_train, y_test = get_data()
    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        sess.run(update_weights, feed_dict = {X_data : x_train, Y_data : y_train})
        loss = sess.run(cost, feed_dict = {X_data : x_train, Y_data : y_train})
        print("epoch = ", epoch, ", cost = ", loss)

