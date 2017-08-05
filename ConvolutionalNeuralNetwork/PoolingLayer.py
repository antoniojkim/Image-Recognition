import numpy as np
from ConvolutionalNeuralNetwork import Layer as CNN

class PoolingLayer(CNN.Layer):
    def __init__(self, F, S):
        self.field = F
        self.stride = S
        self.pool = self.max_pool

    def forward(self, input):

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

        num_out = num_in
        width_out = (width_in - self.field) / self.stride + 1
        height_out = (height_in - self.field) / self.stride + 1
        depth_out = depth_in

        self.max_location = np.zeros((int(num_out), int(width_out), int(height_out), int(depth_out), 2))

        if (width_out % 1) != 0:
            print("Invalid Input Size:  ({W} - {F})/{S}+1 = {O}".format(W=width_in, F=self.field, S=self.stride, O=width_out))
            exit(1)
        elif (height_out % 1) != 0:
            print("Invalid Input Size:  ({H} - {F})/{S}+1 = {O}".format(H=height_in, F=self.field, S=self.stride, O=height_out))
            exit(1)
        else:
            output = np.zeros((int(num_out), int(width_out), int(height_out), int(depth_out)))  # Output Volume
            print("Pooling Output Shape:      {shape}".format(shape=output.shape))

            for n in range(int(num_in)):
                for d in range(int(depth_out)):
                    for h in range(int(height_out)):
                        for w in range(int(width_out)):
                            pool_area = input[n, self.stride * w:(self.field + self.stride * w), self.stride * h:(self.field + self.stride * h), :]
                            pool_value = self.pool(pool_area)
                            output[n, w, h] = pool_value

                            location = list(zip(*np.where(pool_value == pool_area)))
                            location = (location[0][0] + self.stride, location[0][1] + self.stride)

                            self.max_location[n, w, h, d, 0] = location[0]
                            self.max_location[n, w, h, d, 1] = location[1]

            return output

    def max_pool(self, input):
        xShape = input.shape
        if len(xShape) == 3:
            return np.array([np.max(input[:, :, i]) for i in range(xShape[2])])
        return np.max(input)

    def backpropagate(self, output_gradient):

         gradient = np.zeros(self.input.shape)

         for n in range(self.input.shape[0]):
             for d in range(self.input.shape[3]):
                 for h in range(self.input.shape[2]):
                     for w in range(self.input.shape[1]):

                         location = self.max_indices[n, f, w, h, :]
                         gradient[n, f, int(location[0]), int(location[1])] = output_gradient[n, f, w, h]

         return gradient


    def update_parameters(self, learning_rate = 0.5):
        pass