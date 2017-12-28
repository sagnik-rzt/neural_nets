#Builds a single-hidden-layered feedforward neural network classifier for the Iris data set.
#The Iris data-set contains a set of labelled data that is used to classify the 3 types of species
#of Iris flowers. The features are sepal length, sepal width, petal length, and petal width
#(all in centimetres).


import tensorflow as tf
import numpy as np
import pandas as pd


def hypothesis(x, w1, w2, b1, b2):

    #Forward propagation of the input-data tensor 'x' across the layers of neurons
    #Activation function is the sigmoid or logistic function.
    z1 = tf.matmul(x, w1) + b1
    a1 = tf.nn.tanh(z1)
    score = tf.matmul(a1, w2) + b2

    return score


def get_data(train_fraction = 0.7):
    #Returns a randomly generated training and testing set. The samples are drawn out of the
    #iris dataset with replacement.

    data = pd.read_csv(filepath_or_buffer = "iris_dataset.csv")
    data.columns = ['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3']

    train_set = data.take(np.random.permutation(len(data)) [:int((train_fraction * len(data)))])
    x_train, y_train = train_set[['x1', 'x2', 'x3', 'x4']], train_set[['y1', 'y2', 'y3']]

    test_set = data.take(np.random.permutation(len(data)) [:int((1 - train_fraction) * len(data))])
    x_test, y_test = test_set[['x1', 'x2', 'x3', 'x4']], test_set[['y1', 'y2', 'y3']]

    return x_train, x_test, y_train, y_test

def main():

    #Now let's configure the neural network
    #Total 4 input neurons
    input_neurons = 4
    #50 neurons in the hidden layer
    hidden_layers = 6
    #3 classes of Iris flowers, so 3 output neurons.
    output_neurons = 3

    X = tf.placeholder(dtype= tf.float32, shape = [None, input_neurons])
    Y = tf.placeholder(dtype = tf.float32, shape = [None, output_neurons])

    #Weights and biases for the hidden layer
    W1 = tf.Variable(tf.random_normal(shape = [input_neurons, hidden_layers]), dtype = tf.float32)
    B1 = tf.Variable(tf.random_normal(shape=[1, hidden_layers]), dtype=tf.float32)

    #Weights and biases for the output layer
    W2 = tf.Variable(tf.random_normal(shape = [hidden_layers, output_neurons]), dtype = tf.float32)
    B2 = tf.Variable(tf.random_normal(shape = [1, output_neurons]), dtype = tf.float32)

    #Forward propagation
    y_hat = hypothesis(X, W1, W2, B1, B2) #The hypothesis vector
    #Note that here we are not using a softmax function on y_hat since
    # tf.nn.softmax_cross_entropy_with_logits() already does that for us.


    #Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = y_hat))
    update_weights = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

    #Now let's run the tensorflow session
    with tf.Session() as sess:

        x_train, x_test, y_train, y_test = get_data()
        sess.run(tf.global_variables_initializer())
        batch_size = 10
        batches = len(x_train) % batch_size

        for epoch in range(1000):

            for i in range(0, batches * batch_size, batch_size):
            #Perform mini-batch gradient descent for a mini-batch of size 'batch_size'
                sess.run(update_weights, feed_dict = {X : x_train[i : i+batch_size] , Y : y_train[i : i+batch_size]})

            print("epoch = %d, cost = %f"%(epoch, sess.run(cost, feed_dict = {X : x_test, Y : y_test})))

main()