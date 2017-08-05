import numpy as np
import random


class CNN(object):

    def __init__(self, layers):
        if not isinstance(layers, list):
            self.layers = [layers]
        else:
            self.layers = layers

    def feed_forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, training_input, labels, epochs=10, mini_batch_size=50, learning_rate=0.5):
        """
        Stochastic Gradient Descent

        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param learning_rate:
        :return:
        """
        n = len(training_input)
        training_data = list(zip(training_input, labels))

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch)
                print("Epoch #{0} complete. Loss:  {1}".format(i, loss))

    def update_mini_batch(self, mini_batch):
        input, expected_output  = zip(*mini_batch)
        actual_output = self.feed_forward(input)
        output_gradient = self.layers[-1].cost_gradient(expected_output, actual_output)

        for layer in reversed(self.layers):
            output_gradient = layer.backpropagate(output_gradient)

        for layer in self.layers:
            layer.update_parameters(learning_rate)

        return self.layers[-1].cost(expected_output, actual_output)