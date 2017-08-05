import numpy as np
from ConvolutionalNeuralNetwork import Layer as CNN
from NeuralNetworks.NeuralNetwork import NeuralNetwork as NN

class ConvolutionLayer(CNN.Layer):

    def __init__(self, K=4, F=5, S=2, P=0, iD=3):
        self.conv_depth = K
        self.field = F  # Field Size
        self.padding = P
        self.stride = S

        self.inputDepth = iD

        self.weights = np.random.randn(K, F, F, iD)
        self.weights_gradient = np.zeros(self.weights.shape)

    def forward(self, input):
        input = np.array(input)

        self.input = input

        shape = input.shape
        if len(shape) == 2:
            input = np.reshape(input, (1, shape[0], shape[1], self.inputDepth))
            shape = input.shape
        if len(shape) == 3:
            input = np.reshape(input, (shape[0], shape[1], shape[2], self.inputDepth))
            shape = input.shape

        num_in = shape[0]
        width_in = shape[1]
        height_in = shape[2]
        depth_in = shape[3]

        self.num_out = num_in
        self.width_out = (width_in - self.field + 2 * self.padding) / self.stride + 1
        self.height_out = (height_in - self.field + 2 * self.padding) / self.stride + 1
        self.depth_out = self.conv_depth

        if (depth_in != self.inputDepth):
            print("Invalid Input Error:  Depth Size is {D1},   Should be {D2}".format(D1=depth_in, D2=self.inputDepth))
            exit(1)
        elif (self.width_out % 1) != 0:
            print("Invalid Input Size:  ({W} - {F} + 2*{P})/{S}+1 = {O}".format(W=width_in, F=self.field, P=self.padding, S=self.stride, O=width_out))
            exit(1)
        elif (self.height_out % 1) != 0:
            print("Invalid Input Size:  ({H} - {F} + 2*{P})/{S}+1 = {O}".format(H=height_in, F=self.field, P=self.padding, S=self.stride, O=height_out))
            exit(1)
        else:
            output = np.zeros((int(self.num_out), int(self.width_out), int(self.height_out), int(self.depth_out))) # Output Volume
            print("Convolution Output Shape:  {shape}".format(shape=output.shape))

            for n in range(int(self.num_out)):
                for d in range(int(self.depth_out)):
                    for h in range(int(self.height_out)):
                        for w in range(int(self.width_out)):
                            output[n, w, h, d] = np.sum(input[n, self.stride * w:(self.field + self.stride * w), self.stride * h:(self.field + self.stride * h), :] * self.weights[d])

            return output

    def backpropagate(self, output_gradient):

        output_gradient = np.array(output_gradient)

        print(output_gradient.shape)

        input_gradient = np.zeros(self.input.shape)

        for n in range(self.input.shape[0]):
            for d in range(self.input.shape[3]):
                for h in range(0, int(self.height_out), self.stride):
                    for w in range(0, int(self.width_out), self.stride):
                        print(input_gradient[n, w:w+self.field, h:h+self.field, :].shape)
                        print(self.weights[:, :, :, d].shape)
                        input_gradient[n, w:w+self.field, h:h+self.field, :] += output_gradient[n, w, h, d] * self.weights[:, :, :, d]

                for h in range(output_gradient.shape[2]):
                    for w in range(output_gradient.shape[1]):
                        self.weights_gradient[:, :, :, d] += output_gradient[n, w, h, d] * self.input[n, w*self.S:w*self.S+self.field, h*self.S:h*self.S+self.field, :]

        return input_gradient

    def update_parameters(self, learning_rate = 0.5):
        self.weights += self.weights_gradient * learning_rate


    def cost(self, expected_output, actual_output):
        return NN.mean_squared_loss(NN, expected=expected_output, output=actual_output)

    def cost_gradient(self, expected_output, actual_output):
        self.error = NN.mean_squared_loss_prime(NN, expected=expected_output, output=actual_output)

        return self.error
