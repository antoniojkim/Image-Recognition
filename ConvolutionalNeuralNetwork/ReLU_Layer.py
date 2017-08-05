import numpy as np
from ConvolutionalNeuralNetwork import Layer as CNN

class ReLU_Layer(CNN.Layer):

    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    def backpropagate(self, output_gradient):

         input_gradient = np.zeros(self.input.shape)

         input_gradient[self.input >= 0] = 1

         return input_gradient * output_gradient


    def update_parameters(self, learning_rate = 0.5):
        pass