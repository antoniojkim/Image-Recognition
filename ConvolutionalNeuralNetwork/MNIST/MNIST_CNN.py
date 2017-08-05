import numpy as np
import matplotlib.pyplot as plt
import MNIST_Loader as mnist
from ConvolutionalNeuralNetwork.ConvolutionLayer import ConvolutionLayer
from ConvolutionalNeuralNetwork.PoolingLayer import PoolingLayer
from ConvolutionalNeuralNetwork.ReLU_Layer import ReLU_Layer
from ConvolutionalNeuralNetwork.ConvolutionalNN import CNN

# Layers
Convolution1 = ConvolutionLayer(K=5, F=5, S=1, iD=1)
ReLU1 = ReLU_Layer()
PoolingLayer1 = PoolingLayer(F=4, S=2)
Fully_Connected = ConvolutionLayer(F=11, S=1, iD=5, K=10)

network = CNN([Convolution1, ReLU1, PoolingLayer1, Fully_Connected])

training_data, validation_data, test_data = mnist.load_data_wrapper()
training_input, labels  = zip(*training_data)

print("Training Start")
network.train(training_input, labels, mini_batch_size=1)
print("Training End")

def plot_images(self, images):
    if not isinstance(images, list):
        images = [images]
    images_len = len(images)
    plot_sizes = [0, 0]
    while plot_sizes[0]*plot_sizes[1] < images_len:
        plot_sizes[0] += 1
        if plot_sizes[0]*plot_sizes[1] < images_len:
            plot_sizes[1] += 1
        else:
            break
    fig = plt.figure()
    for i in range(images_len):
        ax = fig.add_subplot(plot_sizes[0], plot_sizes[1], i+1)
        ax.matshow(images[i], cmap=plt.get_cmap('Greys'))
    plt.show()