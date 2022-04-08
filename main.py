from mlxtend.data import loadlocal_mnist
import platform
import sys
import numpy as np
import random
import timer

import gzip

import matplotlib.pyplot as plt

"""
Neural Network Design:

Input layer: 
"""

# preset the learning_rate, but will allow us to pass in a learning rate
learning_rate = 0.001

class Neuron:
    def __init__(self) -> None:
        pass

class InputLayer:
    def __init__(self, next_layer = None) -> None:
        self.next_layer = next_layer
    
    def fowardPropagation(self, image, c_label = None, testing = False) -> None:
        self.next_layer.fowardPropagation(image.T, c_label, testing)

    def backwardPropagation(self, backward_pass_data) -> None:
        return

class FullyConnectedLayer:
    def __init__(self, n_neurons, n_incoming_neurons, next_layer = None, previous_layer = None) -> None:
        self.next_layer = next_layer
        self.previous_layer = previous_layer

        self.n_neurons = n_neurons
        self.n_incoming_neurons = n_incoming_neurons

        self.weights = np.random.rand(n_neurons, n_incoming_neurons) - .5
        self.bias = np.random.rand(n_neurons, 1) - .5
        droupoutRateLambda = lambda x: 1/0.6 if x < 0.6 else 0
        self.dropout = droupoutRateLambda(np.random.rand(n_neurons, 1))

        self.avg_weights_deriv = np.zeros((n_neurons, n_incoming_neurons), dtype=float32)
        self.avg_bias_deriv = np.zeros((n_neurons, 1), dtype=float32)
        self.forward_pass_data = None

    def ReLUActivationFunction(self, output):
        return np.maximum(0, output)

    def forwardPropagation(self, forward_pass_data, c_label, testing) -> None:
        self.forward_pass_data = forward_pass_data
        output = ReLUActivationFunction(np.matmul(self.weights, forward_pass_data) + self.bias)
        # output = ReLUActivationFunction(np.matmul(self.weights, forward_pass_data) + self.bias) * self.dropout # can see the performance with dropout
        prediction = self.next_layer.forwardPropagation(output, c_label, testing)
        return prediction

    def backwardPropagation(self, backward_pass_data) -> None:
        downstream_deriv = (np.matmul(backward_pass_data.T, self.weights)).T
        # downstream_deriv = np.matmul((self.dropout * backward_pass_data).T, self.weights)
        self.avg_weights_deriv += np.matmul(backward_pass_data, self.forward_pass_data.T) / 100
        # self.avg_weights_deriv += np.matmul((self.dropout * backward_pass_data), self.forward_pass_data.T) / 100
        self.avg_bias_deriv += (backward_pass_data/100)
        #self.avg_bias_deriv += ((backward_pass_data * self.dropout)/100)
        self.previous_layer(downstream_deriv)

    def updateWeightsAndBias(self, learning_rate) -> None:
        self.weights -= learning_rate * self.avg_weights_deriv
        self.bias -= learning_rate * self.avg_bias_deriv
        self.avg_weights_deriv = np.zeros((n_neurons, n_incoming_neurons), dtype=float32)
        self.avg_bias_deriv = np.zeros((n_neurons, 1), dtype=float32)


class SoftMaxLayer:
    def __init__(self, n_neurons, next_layer = None, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.next_layer = next_layer
        self.previous_layer = previous_layer
        self.forward_pass_output = None
    
    def forwardPropagation(self, forward_pass_data, c_label, testing) -> None:
        """ # This is the formula I had when doing CUDA
        intermediate_output = np.exp(output - max(forward_pass_data))
        output = intermediate_output / sum(intermediate_output)
        """
        self.forward_pass_output = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))
        if not testing:
            self.next_layer.forwardPropagation(self.forward_pass_output, c_label)
        return self.forward_pass_output

    def backwardPropagation(self, backward_pass_data) -> None:
        downstream_deriv = np.zeros((self.n_neurons, 1), dtype=float32)
        for i in range(self.n_neurons):
            accum = backward_pass_data[i][0] * self.forward_pass_output[i] * (1 - self.forward_pass_output[i])
            forward_pass_output_remove_ith_column = np.delete(self.forward_pass_output, i, 1)
            backward_pass_data_remove_ith_column = np.delete(self.backward_pass_data, i, 0).T
            accum += forward_pass_output_remove_ith_column.dot(backward_pass_data_remove_ith_column)
            downstream_deriv[i][0] += accum
        return downstream_deriv
        

class CrossEntropyLayer:
    def __init__(self, n_neurons, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.previous_layer = previous_layer

    def forwardPropagation(self, forward_pass_data, c_label) -> None:
        downstream_deriv = np.zeros((n_neurons, 1), dtype=float32)
        downstream_deriv[c_label][0] = -1/forward_pass_data[c_label][0]
        self.previous_layer.backwardPropagation(downstream_deriv)

def read_int(file_pointer: gzip.GzipFile) -> int:  
    """ Returns the int of the first 4 bytes
    """
    return int.from_bytes(file_pointer.read(4), "big")

def read_training_data(training_image_path: str, training_label_path: str) -> None:
    """ Returns a numpy array of the training images and labels
    """
    return loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte')

def read_training_data_slow(training_image_path: str, training_label_path: str) -> None:
    # Process the training images
    with gzip.open(training_image_path, 'r') as reader:
        # Read the meta data 
        reader.read(4) # Read past the magic number
        n_images, n_rows, n_cols = read_int(reader), read_int(reader), read_int(reader)

        training_images = np.zero((n_rows, n_cols, n_images), dtype=np.float16)

        for i in range(n_images):
            buf = reader.read(n_rows * n_cols)
            tmp_image = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, n_cols)
            for row in range(n_rows):
                for col in range(n_cols):
                    # Normalize the values from -1 to 1
                    training_images[row][col][i] = tmp_image[row][col]/127.5 - 1
    
    print('Finishing processing training images')

    # Process the training labels
    with gzip.open(training_label_path, 'r') as reader:
        # Read the meta data 
        reader.read(4) # Read past the magic number
        n_labels = read_int(reader)
        buf = reader.read(n_labels)
        training_labels = np.frombuffer(buf, dtype=np.uint8)

    print('Finishing processing training labels')

    return training_images, training_labels

def connect_layers(list_layers) -> None:
    for i in range(len(list_layers)):
        if i - 1 > -1:
            list_layers[i].previous_layer = list_layers[i-1]

        if i + 1 < len(list_layers):
            list_layers[i].next_layer = list_layers[i+1]


def train_neural_network(training_images, training_labels, epochs) -> None:
    # Construct the Neural Network
    input_layer = InputLayer()
    first_fully_connected_layer = FullyConnectedLayer(1024, 784)
    output_layer = FullyConnectedLayer(10, 1024)
    softmax_layer = SoftMaxLayer(10)
    cross_entropy_layer = CrossEntropyLayer(10)
    list_layers = [input_layer, first_fully_connected_layer, output_layer, softmax_layer, cross_entropy_layer]

    # Connect the layers
    connect_layers(list_layers)


    for epoch in range(epochs):
        print(f"\n----------------------- STARTING EPOCH {epoch} -------------------\n";
        start_time = time.perf_counter()

        for c_round in range(600):
            # test the neural network every 100 rounds for each epoch
            if c_round % 100 == 0:
                corrects = 0
                for iter in range(10000):
                    random_test = random.randint(0, training_images.shape[0])
                    prediction = input_layer.fowardPropagation(training_images[index], None, True)
                    if prediction == training_labels[random_test]: corrects += 1
                print(f"Epoch {epoch}: Round {c_round:5d}: accuracy={(correct/10000.0):0.6f}\n")
            
            # continue to train the neural network
            for iter in range(100):
                index = (c_round * 100) + iter
                input_layer.fowardPropagation(training_images[index], training_labels[index])

        end_time = time.perf_counter()
        print(f"This epoch took: {(end_time - start_time):0.6f}\n")






def main() None:
    """
    if (len(sys.argv) < 5):
        print("Too few args. Pass in ./program training_image_file training_label_file test_image_file test_label_file")
        return 
    """
    
    training_image_path, training_label_path, epochs = sys.argv[1], sys.argv[2], 20

    # Read in training data
    training_images, training_labels = read_training_data(training_image_path, training_label_path)
    print("Finished reading all the images into the program.")

    # Train the Neural Network
    train_neural_network(training_images, training_labels)

    # Test the Neural Network

if __name__ == "__main__":
    main()
