from mlxtend.data import loadlocal_mnist
import platform
import sys
import numpy as np
import random
import time
from sklearn.preprocessing import OneHotEncoder

import gzip

import matplotlib.pyplot as plt

"""
Passes through preset batch size

Can try 2 things:
1. Pass in an entire train_data (both train_images and train_labels --> permutate it) * trying this one first
2. Pass in seperate train_images and train_labels --> permuate the indicies and create each batch
"""

# learning_rate - learning rate of the nueral network
# batch_size - is the number of training images before preforming gradient descent
learning_rate = 0.001
batch_size = 100
train_images, train_labels = None, None

# timing how long each operation takes
accum_time_FCL1_forward_prop = 0
accum_time_FCL2_forward_prop = 0
accum_time_FCL1_backward_prop = 0
accum_time_FCL2_backward_prop = 0
accum_time_SML_forward_prop = 0
accum_time_SML_backward_prop = 0
accum_time_CEL_forward_prop = 0

class Neuron:
    def __init__(self) -> None:
        pass

# Try passing in the entire training and testing dataset
class InputLayer:
    def __init__(self, train_data, next_layer = None) -> None:
        # train_data - index 0 is image, and index 1 is the label
        self.train_data = train_data
        self.train_data_index = 0       # increment this index whenever we call the forward prop
        self.next_layer = next_layer

    # shuffle the the training data
    def shuffle_training_data(self) -> None:
        self.train_data = np.random.permutation(self.train_data)
        self.train_data_index = 0
    
    # returns the prediction[s] of the input
    # training - pass in the (shuffled) train images to the next layer
    # testing - pass in the testing images to the next layer
    def forward_propagation(self, test_image = None, is_training = True) -> int:
        
        if is_training:
            global train_images, train_labels
            train_data = self.train_data[self.train_data_index : self.train_data_index + batch_size].T
            train_images, train_labels = train_data[1:], train_data[0]
            self.train_data_index += batch_size

        return self.next_layer.forward_propagation(train_images, is_training) if is_training else self.next_layer.forward_propagation(test_image, is_training)


    # simply just returnt to the previous layer
    def backward_propagation(self, backward_pass_data) -> int:
        return -1

class FullyConnectedLayer:
    def __init__(self, n_neurons, n_incoming_neurons, is_output_layer=False, next_layer = None, previous_layer = None) -> None:
        # connect the layers
        self.next_layer = next_layer
        self.previous_layer = previous_layer

        # data about the layer
        self.n_neurons = n_neurons
        self.is_output_layer = is_output_layer
        self.n_incoming_neurons = n_incoming_neurons

        # initialize weights, bias, and dropout (forward prop)
        self.weights = np.random.rand(n_neurons, n_incoming_neurons) - .5 # (1024, 784)
        self.bias = np.random.rand(n_neurons, 1) - .5
        self.dropout = np.array(list(map(lambda x: 1/0.6 if x < 0.6 else 0, np.random.rand(n_neurons)))).reshape((n_neurons, 1))

        # ! possibly don't need this, because everytime we are going to just
        # self.avg_weights_deriv = None
        # self.avg_bias_deriv = None
        self.forward_pass_data = None

    def ReLU_activation_function(self, output):
        return np.maximum(0, output)

    # is_training - True, then modify the weights and bias
    #             - False, then return an array of predictions 
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        global accum_time_FCL1_forward_prop 
        global accum_time_FCL2_forward_prop

        start_time = time.perf_counter()

        self.forward_pass_data = forward_pass_data # (784, 100)
        output = self.weights.dot(forward_pass_data) + self.bias 
        if not self.is_output_layer:
            output = self.ReLU_activation_function(output) * self.dropout

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if is_training:
            if self.is_output_layer: accum_time_FCL2_forward_prop += total_time
            else: accum_time_FCL1_forward_prop += total_time
        
        return self.next_layer.forward_propagation(output, is_training)

    # ! Need to check the upstream_gradient computation
    def backward_propagation(self, backward_pass_data) -> None:
        upstream_gradient, avg_weights_deriv, avg_bias_deriv = None, None, None
        if self.is_output_layer:
            upstream_gradient = np.matmul(self.weights.T, backward_pass_data)
            avg_weights_deriv = np.sum(np.matmul(backward_pass_data, self.forward_pass_data.reshape((batch_size, 1, self.n_incoming_neurons))), axis=0)
            avg_bias_deriv = np.sum(backward_pass_data, axis = 0)
        else:
            upstream_gradient = np.matmul((self.dropout * backward_pass_data).T, self.weights) 
            avg_weights_deriv = np.sum(np.matmul((self.dropout * backward_pass_data), self.forward_pass_data.reshape((batch_size, 1, self.n_incoming_neurons))), axis=0)
            avg_bias_deriv = np.sum(backward_pass_data * self.dropout, axis = 0)
        
        # update the weights and bias 
        self.weights -= learning_rate * self.avg_weights_deriv
        self.bias -= learning_rate * self.avg_bias_deriv

        self.previous_layer.backward_propagation(upstream_gradient)

    # ! Not using this anymore
    def update_weights_and_bias(self, learning_rate) -> None:
        self.weights -= learning_rate * self.avg_weights_deriv
        self.bias -= learning_rate * self.avg_bias_deriv
        self.avg_weights_deriv = None
        self.avg_bias_deriv = None


class SoftMaxLayer:
    def __init__(self, n_neurons, next_layer = None, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.next_layer = next_layer
        self.previous_layer = previous_layer
        self.forward_pass_output = None # softmax - (10, 100)

    def extract_prediction(self):
        return self.forward_pass_output.argmax(axis = 0)
    
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        """ # This is the formula I had when doing CUDA
        # self.forward_pass_output = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))
        # print(self.forward_pass_output)/
        """

        global accum_time_SML_forward_prop
        start_time = time.perf_counter()

        self.forward_pass_output = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))

        end_time = time.perf_counter()
        total_time = end_time - start_time
        if is_training:
            accum_time_SML_forward_prop += total_time

        return self.next_layer.forward_propagation(self.forward_pass_output) if is_training else self.extract_prediction()

    # backward_pass_data - (10,100)
    def backward_propagation(self, backward_pass_data) -> None:
        global accum_time_SML_backward_prop
        start_time = time.perf_counter()
        
        upstream_gradient = np.zeros((self.n_neurons, batch_size))
        softmax = self.forward_pass_output.T # (100, 10)
        gradient = backward_pass_data.T # (100, 10)

        print(f"softmax: {softmax}")
        print(f"gradient: {gradient}")

        """
[2.37346418e-10 2.75264834e-11 9.99970009e-01 4.60658020e-12 2.62061065e-09 3.71386098e-08 2.52849346e-08 2.49914572e-05 1.84102599e-12 4.93387848e-06]

        """

        for train_index in range(batch_size):
            softmax_ex = softmax[train_index] # (1, 10)
            d_softmax_ex  = softmax_ex * np.identity(softmax_ex.size) - softmax_ex.T @ softmax_ex # (10, 10)

            upstream_gradient_ex = d_softmax_ex @ gradient[train_index] # (10, 1)
            upstream_gradient[np.arange(self.n_neurons), train_index] = upstream_gradient_ex

            for i in range(self.n_neurons):
                if upstream_gradient_ex[i] != upstream_gradient[i][train_index]:
                    print(f"Incorrect: upstream: {upstream_gradient[i][train_index]}, my_comp: {upstream_gradient_ex[i]}")
        
        print("upstream_gradient", upstream_gradient)

        #! try to optimize this if possible
        # (100, 10, 1)
        """ 
        accum = backward_pass_data[neuron][0] * self.forward_pass_output[neuron][0] * (1 - self.forward_pass_output[neuron][0])
        accum += (backward_pass_data.T.dot(self.forward_pass_output) * -self.forward_pass_output[neuron][0]) - (backward_pass_data[neuron][0] * self.forward_pass_output[neuron][0] * -self.forward_pass_output[neuron][0])
        """
        num_incorrect, train_index_incorrect = 0, []

        downstream_deriv = np.zeros((self.n_neurons, batch_size))
        for train_index_2 in range(batch_size):
            for neuron in range(self.n_neurons):
                accum = (gradient[train_index_2].dot(softmax[train_index_2]  * -softmax[train_index_2][neuron]))

                # adjust for when neuron of SML = neuron of CEL
                accum -= (gradient[train_index_2][neuron] * softmax[train_index_2][neuron] * -softmax[train_index_2][neuron])
                accum += gradient[train_index_2][neuron] * softmax[train_index_2][neuron] * (1 - softmax[train_index_2][neuron])
                downstream_deriv[neuron][train_index_2] = accum

                if accum != upstream_gradient[neuron][train_index_2]:
                    print(f"Incorrect: {accum} {upstream_gradient[neuron][train_index_2]}")
 

        print("downstream", downstream_deriv)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        accum_time_SML_backward_prop += total_time
        # print(f'SoftMaxLayer Back Prop took: {end_time - start_time}')
        # self.previous_layer.backward_propagation(upstream_gradient)
        

class CrossEntropyLayer:
    def __init__(self, n_neurons, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.previous_layer = previous_layer

    # ensure that the upstream_gradient has shape (10, batch_size)
    def validate_upstream_gradient(self, upstream_gradient) -> None:
        train_labels_set = set(train_labels)
        for num in range(10):
            if num not in train_labels_set:
                upstream_gradient = np.insert(upstream_gradient, num, 0, axis = 0)
                

    # forward_pass_data - (10, batch_size) - every column is a training data, and every row represents a label
    # upstream_gradient - (10, batch_size)
    def forward_propagation(self, forward_pass_data) -> None:
        global accum_time_CEL_forward_prop
        start_time = time.perf_counter()

        one_hot_encoder = OneHotEncoder(sparse = False)
        upstream_gradient = one_hot_encoder.fit_transform(train_labels.reshape(batch_size, 1)).T * (-1/forward_pass_data)
        # need to ensure that the upstream_gradient.shape === (10, batch_size) (if fewer than there is a number that is never shown in train_labels)
        self.validate_upstream_gradient(upstream_gradient)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        accum_time_CEL_forward_prop += total_time

        """
        for index, label in enumerate(train_labels):
            label = int(label)
            if label < 0 or label > 10:
                print("Label is wrong")
            # print(forward_pass_data[label][index], -1/forward_pass_data[label][index], upstream_gradient[label][index])
            if (forward_pass_data[label][index] == 0) or (-1/forward_pass_data[label][index] != upstream_gradient[label][index]):
                print(forward_pass_data[index])
                print("WTF is going on here")
        # print(train_labels[0])
        # print(upstream_gradient[0])
        """

        self.previous_layer.backward_propagation(upstream_gradient)

""" --- Reading in training and testing images --- """
# def normalize_images(images) -> np.ndarray:
#     normalizerLambda = lambda x: (x/127.5) - 1
#     return np.array(list(map(normalizerLambda, images)))

def read_training_data() -> (np.ndarray, np.ndarray):
    """ Returns a numpy array of the training images and labels
    """
    training_images, training_labels = loadlocal_mnist(
        images_path='./mnist/train-images-idx3-ubyte', 
        labels_path='./mnist/train-labels-idx1-ubyte')
    return training_images/255, training_labels

def read_testing_data() -> (np.ndarray, np.ndarray):
    testing_images, testing_labels = loadlocal_mnist(
        images_path='./mnist/t10k-images-idx3-ubyte', 
        labels_path='./mnist/t10k-labels-idx1-ubyte')
    return testing_images/255, testing_labels

""" --- Training the NN --- """
def print_current_metrics(epoch) -> None:
    print("Some stats:")
    print(f"avg_time_FCL1_forward_prop: {accum_time_FCL1_forward_prop/(epoch * 60000)}")
    print(f"avg_time_FCL2_forward_prop: {accum_time_FCL2_forward_prop/(epoch * 60000)}")
    print(f"avg_time_FCL1_backward_prop: {accum_time_FCL1_backward_prop/(epoch * 60000)}")
    print(f"avg_time_FCL2_backward_prop: {accum_time_FCL2_backward_prop/(epoch * 60000)}")
    print(f"avg_time_SML_forward_prop: {accum_time_SML_forward_prop/(epoch * 60000)}")
    print(f"avg_time_SML_backward_prop: {accum_time_SML_backward_prop/(epoch * 60000)}")
    print(f"avg_time_CEL_forward_prop: {accum_time_CEL_forward_prop/(epoch * 60000)}")
    print()

def connect_layers(list_layers) -> None:
    for i in range(len(list_layers)):
        if i - 1 > -1: list_layers[i].previous_layer = list_layers[i-1]
        if i + 1 < len(list_layers): list_layers[i].next_layer = list_layers[i+1]


def train_neural_network(train_images, train_labels, train_data, epochs, n_neurons, learning_rate = 0.001) -> None:
    global batch_size

    # Construct the Neural Network
    input_layer = InputLayer(train_data)
    first_fully_connected_layer = FullyConnectedLayer(n_neurons, 784)
    output_layer = FullyConnectedLayer(10, n_neurons, True)
    softmax_layer = SoftMaxLayer(10)
    cross_entropy_layer = CrossEntropyLayer(10)

    list_layers = [input_layer, first_fully_connected_layer, output_layer, softmax_layer, cross_entropy_layer]

    # Connect the layers
    connect_layers(list_layers)

    for epoch in range(epochs):

        # shuffle the training images and labels 
        input_layer.shuffle_training_data()

        print(f"\n----------------------- STARTING EPOCH {epoch} -------------------\n")
        start_time = time.perf_counter()

        for c_round in range(1):
            # test the neural network every 100 rounds for each epoch
            # ! Need to find a way to test the data efficently; pass the entire train in as a arguement
            if c_round % 100 == 0:
                batch_size, n_corrects = 1, 0
                for iter in range(1):
                    random_test = random.randint(0, train_images.shape[0]-1)
                    test_image = train_images[random_test].reshape((784, 1))
                    prediction = input_layer.forward_propagation(test_image, False)
                    if prediction == train_labels[random_test]: n_corrects += 1
                print(f"Epoch {epoch}: Round {c_round:5d}: accuracy={(n_corrects/10000.0):0.6f}")
                batch_size = 100
                
            # train the neural network
            input_layer.forward_propagation()

        end_time = time.perf_counter()
        print(f"This epoch took: {(end_time - start_time):0.6f}\n")
        print_current_metrics(epoch+1)

    return input_layer





def main() -> None:
    """
    if (len(sys.argv) < 5):
        print("Too few args. Pass in ./program training_image_file training_label_file test_image_file test_label_file")
        return 
    """
    
    # training_image_path, training_label_path, epochs = sys.argv[1], sys.argv[2], 20
    epochs, n_neurons = 1, 128
    
    start = time.perf_counter()
    # Read in training data
    print(f"Starting to read training and testing data for the neural network with {n_neurons} neurons")
    train_images, train_labels = read_training_data()
    train_data = np.append(train_labels.reshape((1, len(train_labels))), train_images.T, axis = 0).T # each row is a training data

    test_images, test_labels = read_testing_data()
    # test_data = np.append(test_labels.reshape((1, len(test_labels))), test_images.T, axis = 0).T # each row is a testing data
    end = time.perf_counter()
    print(f"Finished reading all the images into the program, in {end - start} seconds")

    # Train the Neural Network
    print("Starting to train the neural network")
    neural_network = train_neural_network(train_images, train_labels, train_data, epochs, n_neurons)
    print("Finish training the neural network")

    """
    # Test the Neural Network
    print("\nStarting to test the neural network on testing data")
    test_neural_network(testing_images, testing_labels, neural_network)
    print("Finish testing the neural network")
    """

if __name__ == "__main__":
    main()