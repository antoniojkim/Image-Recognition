import numpy as np

class FixedNN(object):
    """
    This is a simple neural network with 1 hidden layer.
    There are 3 neurons in the input layer, 4 neurons in the hidden layer and 1 neurons in the output layer.
    """

    def __init__(self):

        # Hyperparameters
        self.inputSize = 3  # D
        self.hiddenSize1 = 4 # H
        self.outputSize = 1 # K

        # Regularization Strength
        self.reg = 0.01

        self.weights1 = np.random.randn(self.inputSize, self.hiddenSize1)   # D x H matrix
        print(self.weights1.shape)
        self.weights2 = np.random.randn(self.hiddenSize1, self.outputSize)  # H x K matrix
        print(self.weights2.shape)

    def feed_forward(self, x):
        # x = input:  M x D matrix
        self.z2 = np.dot(x, self.weights1)
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        output = self.activation(self.z3)
        return output

    def activation(self, z):
        """
        The sigmoid function.

        :param z:  the input for the function.
        :param w:  an optional weight that will be multiplied to z
        :param b:  an optional bias that will be added to z
        :return:  the value produced by applying the activation function to z.
        """
        return 1.0/(1+np.exp(-z))
    def activation_Prime(self, z):
        """
        The derivative of the sigmoid function.

        :param z: The input for the function
        :return:  The value produced by applying the derivative of the activation function to z
        """
        activation = self.activation(z)
        return activation*(1-activation)

    def cost_function(self, output, expected):
        return 0.5*np.sum((output-expected)**2)

    def train(self, training_input, expected_output, eta):

        training_input = np.array(training_input)
        expected_output = np.array(expected_output)

        output = self.feed_forward(training_input)

        loss = self.cost_function(output, expected_output)

        delta3 = np.multiply(output-expected_output, self.activation_Prime(self.z3))
        dWeights2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.weights2.T) * self.activation_Prime(self.z2)
        dWeights1 = np.dot(training_input.T, delta2)

        self.weights1 = self.weights1 - eta * dWeights1
        self.weights2 = self.weights2 - eta * dWeights2

        return loss

training_input = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
expected_output = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

nn = FixedNN()

print(nn.feed_forward(training_input))

for i in range(1, 2001):
    loss = nn.train(training_input, expected_output, 0.5)
    if i%200 == 0:
        print("Epoch", i, ":   Loss: ", loss)

print(nn.feed_forward(training_input))