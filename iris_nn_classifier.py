#Builds a single-hidden-layered feedforward neural network classifier for the Iris data set.
#The Iris data-set contains a set of labelled data that is used to classify the 3 types of species
#of Iris flowers. The features are sepal length, sepal width, petal length, and petal width
#(all in centimetres).


import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

random_seed = 6
tf.set_random_seed(random_seed)

def forward_prop(x, w1, w2):
    #Forward propagation of the input-data tensor 'x' across the layers of neurons
    #Activation function is the sigmoid or logistic function.

    z1 = tf.matmul(x, w1)
    a1 = tf.nn.tanh(z1)
    score = tf.matmul(a1, w2)
    return score


def get_data():
    #Loads the Iris dataset and then splits them into training and test sets.

    iris = datasets.load_iris()
    x_data = iris["data"]
    y_target = iris["target"]

    #Append a column full of 1s to the feature vector for the bias term
    m, n = x_data.shape
    X = np.ones((m, n + 1))
    X[:, 1:] = x_data

    #Convert target values to one-hot vectors
    num_label_types = len(np.unique(y_target))
    Y = np.eye(num_label_types)[y_target]

    #Use 30% of the data as testing data and the rest as training data
    return train_test_split(X, Y, test_size = 0.30, random_state = random_seed)


def main():

    #Now let's configure the neural network
    #Total 5 input neurons; 4 feature neurons and 1 bias neuron
    input_neurons = 5
    #50 neurons in the hidden layer
    hidden_layers = 50
    #3 classes of Iris flowers, so 3 output neurons.
    output_neurons = 3

    X = tf.placeholder(dtype= tf.float32, shape = [None, input_neurons])
    Y = tf.placeholder(dtype = tf.float32, shape = [None, output_neurons])

    W1 = tf.Variable(tf.random_normal(shape = [input_neurons, hidden_layers]), dtype = tf.float32)
    W2 = tf.Variable(tf.random_normal(shape = [hidden_layers, output_neurons]), dtype = tf.float32)

    #Forward propagation
    y_hat = forward_prop(X, W1, W2) #The hypothesis vector
    y_predict = tf.cast(tf.argmax(y_hat, axis = 1), tf.float32)  #Getting the final score from the hypothesis vector

    #Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = y_hat))
    update_weights = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

    #Now let's run the tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 14

        for epoch in range(1000):
            x_train, x_test, y_train, y_test = get_data()

            for i in range(len(x_train)):
            #Perform mini-batch gradient descent for a mini-batch of size 'batch_size'
                sess.run(update_weights, feed_dict = {X : x_train[i : i+batch_size] , Y : y_train[i : i+batch_size]})

            training_accuracy = np.mean(np.argmax(y_train, axis = 1) == sess.run(y_predict, feed_dict = {X : x_train, Y : y_train}))
            testing_accuracy =  np.mean(np.argmax(y_test, axis = 1) == sess.run(y_predict, feed_dict = {X : x_test, Y : y_test}))

            print("epoch = %d, training_accuracy = %f, testing_accuracy = %f"%(epoch, training_accuracy, testing_accuracy))

main()