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
        self.train_data = np.random.permutation(train_data)
        self.train_data_index = 0       # increment this index whenever we call the forward prop
        self.next_layer = next_layer

    # shuffle the the training data
    def shuffle_training_data(self) -> None:
        # start = time.perf_counter()
        # self.train_data = np.random.permutation(self.train_data) # 0.23565608299999852
        np.random.shuffle(self.train_data) # 0.1674007109999991
        self.train_data_index = 0
        # end = time.perf_counter()
        # print("shuffle_training_data took: ", end - start)
    
    # returns the prediction[s] of the input
    # training - pass in the (shuffled) train images to the next layer
    # testing - pass in the testing images to the next layer
    def forward_propagation(self, test_image = None, is_training = True) -> int:
        
        if is_training:
            global train_images, train_labels
            train_data = self.train_data[self.train_data_index : self.train_data_index + batch_size].T
            train_images, train_labels = train_data[1:], train_data[0].astype(np.int32)
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
        self.weights = np.random.rand(n_neurons, n_incoming_neurons) - .5 # (n, 784) or (10. n)
        self.bias = np.random.rand(n_neurons, 1) - .5
        self.dropout = None

        # ! possibly don't need this, because everytime we are going to just
        # self.avg_weights_deriv = None
        # self.avg_bias_deriv = None
        self.forward_pass_data = None

    def generate_dropout(self):
        droupoutRateLambda = lambda x: 1/0.6 if x < 0.6 else 0
        self.dropout = np.array(list(map(droupoutRateLambda, np.random.rand(self.n_neurons)))).reshape((self.n_neurons,1))

    def ReLU_activation_function(self, output):
        return np.maximum(0, output)

    def ReLU_gradient(self, upstream_gradient):
        return upstream_gradient > 0

    # is_training - True, then modify the weights and bias
    #             - False, then return an array of predictions 
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        global accum_time_FCL1_forward_prop 
        global accum_time_FCL2_forward_prop

        start_time = time.perf_counter()

        self.forward_pass_data = forward_pass_data # (784, 100)
        output = self.weights.dot(forward_pass_data) + self.bias 
        if not self.is_output_layer:
            output = self.ReLU_activation_function(output) 
            if is_training:
                output *= self.dropout

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if is_training:
            if self.is_output_layer: accum_time_FCL2_forward_prop += total_time
            else: accum_time_FCL1_forward_prop += total_time
        
        return self.next_layer.forward_propagation(output, is_training)

    # ! Need to check the upstream_gradient computation
    # FCL1: upstream_gradient - (n, 100); forward_pass_data - (784, 100)
    # FCL2: upstream_gradient  - (10, 100); forward_pass_data - (n, 100) --> (10, n)
    def backward_propagation(self, upstream_gradient) -> None:
        global accum_time_FCL1_backward_prop, accum_time_FCL2_backward_prop
        start_time = time.perf_counter()
        # print(self.forward_pass_data.shape)
        downstream_gradient, avg_weights_deriv, avg_bias_deriv = None, None, None
        if self.is_output_layer:
            # downstream_gradient = np.matmul(self.weights.T, upstream_gradient)

            downstream_gradient = self.weights.T @ upstream_gradient
            # downstream_gradient = upstream_gradient

            # avg_weights_deriv = np.sum(np.matmul(upstream_gradient, self.forward_pass_data.reshape((batch_size, 1, self.n_incoming_neurons))), axis=0)
            # avg_weights_deriv = np.sum(upstream_gradient @ self.forward_pass_data.T, axis = 0) / batch_size
            avg_weights_deriv = (upstream_gradient @ self.forward_pass_data.T) / batch_size
            avg_bias_deriv = np.sum(upstream_gradient, axis = 1).reshape(self.n_neurons, 1) / batch_size
        else:
            
            # downstream_gradient = np.matmul((self.dropout * upstream_gradient).T, self.weights) 
            downstream_gradient = self.weights.T @ (self.dropout * upstream_gradient) * self.ReLU_gradient(self.forward_pass_data)
            # avg_weights_deriv = np.sum(np.matmul((self.dropout * upstream_gradient), self.forward_pass_data.reshape((batch_size, 1, self.n_incoming_neurons))), axis=0)


            # avg_weights_deriv = (self.dropout * upstream_gradient) @ self.forward_pass_data.T / batch_size
            avg_weights_deriv = ((self.dropout * upstream_gradient) @ self.forward_pass_data.T) / batch_size

            # avg_bias_deriv = np.sum(self.dropout * upstream_gradient, axis = 1).reshape(self.n_neurons, 1) / batch_size
            avg_bias_deriv = np.sum(self.dropout * upstream_gradient, axis = 1).reshape(self.n_neurons, 1) / batch_size
        
        # Update the weights and bias 
        # if self.weights.shape != avg_weights_deriv.shape:
        #     print("Incorrect", self.weights.shape, avg_weights_deriv.shape)
        # if self.bias.shape != avg_bias_deriv.shape:
        #     print("Incorrect 2")
        self.weights -= learning_rate * avg_weights_deriv
        self.bias -= learning_rate * avg_bias_deriv

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if self.is_output_layer:
            accum_time_FCL2_backward_prop += total_time
        else:
            accum_time_FCL1_backward_prop += total_time

        self.previous_layer.backward_propagation(downstream_gradient)

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
        self.softmax = None # softmax - (10, 100)

    def extract_prediction(self):
        return self.softmax.argmax(axis = 0)

    # ensure that the upstream_gradient has shape (10, batch_size)
    def validate_upstream_gradient(self, upstream_gradient) -> None:
        train_labels_set = set(train_labels)
        for num in range(10):
            if num not in train_labels_set:
                upstream_gradient = np.insert(upstream_gradient, num, 0, axis = 0)
    
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        """ # This is the formula I had when doing CUDA
        # self.softmax = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))
        # print(self.softmax)/
        """

        global accum_time_SML_forward_prop
        start_time = time.perf_counter()

        self.softmax = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))

        end_time = time.perf_counter()
        total_time = end_time - start_time
        if is_training:
            accum_time_SML_forward_prop += total_time

        return self.next_layer.forward_propagation(self.softmax) if is_training else self.extract_prediction()

    def backward_propagation(self) -> None:
        global accum_time_SML_backward_prop
        start_time = time.perf_counter()

        # one_hot_encoder = OneHotEncoder(sparse = False)
        # one_hot_enocode = one_hot_encoder.fit_transform(train_labels.reshape(batch_size, 1)).T 
        # # need to ensure that the upstream_gradient.shape === (10, batch_size) (if fewer than there is a number that is never shown in train_labels)
        # self.validate_upstream_gradient(one_hot_enocode)
        # upstream_gradient = self.softmax - one_hot_enocode
        # print(upstream_gradient.shape)

        # upstream_gradient2 = self.softmax[train_labels, np.arange(len(train_labels))] - 1
        self.softmax[train_labels, np.arange(batch_size)] = self.softmax[train_labels, np.arange(batch_size)] - 1
        
        # for i in range(len(upstream_gradient)):
        #     for j in range(len(upstream_gradient[0])):
        #         if self.softmax[i][j] != upstream_gradient[i][j]:
        #             print("computation is fucked somewhere")


        # upstream_gradient = self.softmax[train_labels, np.arange(len(train_labels))] - 1
        """
        upstream_gradient = np.zeros((self.n_neurons, batch_size))
        softmax = self.forward_pass_output.T # (100, 10)
        gradient = backward_pass_data.T # (100, 10)

        print(f"softmax: {softmax}")
        print(f"gradient: {gradient}")

        for train_index in range(batch_size):
            softmax_ex = softmax[train_index] # (1, 10)
            d_softmax_ex  = (softmax_ex * np.identity(softmax_ex.size)) - (softmax_ex.T @ softmax_ex) # (10, 10)

            upstream_gradient_ex = d_softmax_ex @ gradient[train_index] # (10, 1)
            upstream_gradient[np.arange(self.n_neurons), train_index] = upstream_gradient_ex

            for i in range(self.n_neurons):
                if upstream_gradient_ex[i] != upstream_gradient[i][train_index]:
                    print(f"Incorrect: upstream: {upstream_gradient[i][train_index]}, my_comp: {upstream_gradient_ex[i]}")
        
        print("upstream_gradient", upstream_gradient)
        """

        #! try to optimize this if possible
        # (100, 10, 1)
        """ 
        accum = backward_pass_data[neuron][0] * self.forward_pass_output[neuron][0] * (1 - self.forward_pass_output[neuron][0])
        accum += (backward_pass_data.T.dot(self.forward_pass_output) * -self.forward_pass_output[neuron][0]) - (backward_pass_data[neuron][0] * self.forward_pass_output[neuron][0] * -self.forward_pass_output[neuron][0])
        """
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
        """
        end_time = time.perf_counter()
        total_time = end_time - start_time
        accum_time_SML_backward_prop += total_time
        # print(f'SoftMaxLayer Back Prop took: {end_time - start_time}')
        self.previous_layer.backward_propagation(self.softmax)
        

class CrossEntropyLayer:
    def __init__(self, n_neurons, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.previous_layer = previous_layer

    # softmax - (10, batch_size) - every column is a training data, and every row represents a label
    # upstream_gradient - (10, batch_size)
    def forward_propagation(self, softmax) -> None:
        global accum_time_CEL_forward_prop
        start_time = time.perf_counter()

        """
        one_hot_encoder = OneHotEncoder(sparse = False)
        upstream_gradient = one_hot_encoder.fit_transform(train_labels.reshape(batch_size, 1)).T * (-1/forward_pass_data)
        # need to ensure that the upstream_gradient.shape === (10, batch_size) (if fewer than there is a number that is never shown in train_labels)
        self.validate_upstream_gradient(upstream_gradient)
        """
        
        # computes the loss for each training entry, but it's not needed as it gets cancelled out when doing the gradient of softmax
        # loss = -np.log(softmax[train_labels, np.arange(len(train_labels))]) 
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        accum_time_CEL_forward_prop += total_time

        self.previous_layer.backward_propagation()

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
    return training_images/255, training_labels.astype(np.int32)

def read_testing_data() -> (np.ndarray, np.ndarray):
    testing_images, testing_labels = loadlocal_mnist(
        images_path='./mnist/t10k-images-idx3-ubyte', 
        labels_path='./mnist/t10k-labels-idx1-ubyte')
    return testing_images/255, testing_labels.astype(np.int32)

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


def generate_random_test(train_data, num_tests):
    # start = time.perf_counter()
    permutated_train_data = np.random.permutation(train_data[:num_tests]).T
    test_images, test_labels = permutated_train_data[1:], permutated_train_data[0].astype(np.int32)
    # end = time.perf_counter()
    # print("generate_random_test took: ", end - start)
    return test_images, test_labels

def train_neural_network(train_images, train_labels, train_data, test_images, test_labels, epochs, n_neurons, learning_rate = 0.001) -> None:
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

        for c_round in range(600):
            # test the neural network every 100 rounds for each epoch
            # ! Need to find a way to test the data efficently; pass the entire train in as a arguement
            if c_round % 100 == 0:
                
                validation_images, validation_labels = generate_random_test(train_data, 10000)
                
                prediction = input_layer.forward_propagation(validation_images, False)
                n_corrects = np.sum(prediction == validation_labels)
                
                print(f"Epoch {epoch}: Round {c_round:5d}: accuracy={(n_corrects/10000.0):0.6f} number of corrects={n_corrects}")

                # batch_size, n_corrects = 1, 0
                # for iter in range(10000):
                #     random_test = random.randint(0, train_images.shape[0]-1)
                #     test_image = train_images[random_test].reshape((784, 1))
                #     prediction = input_layer.forward_propagation(test_image, False)
                #     if prediction[0] == train_labels[random_test]: n_corrects += 1
                # print(f"Epoch {epoch}: Round {c_round:5d}: accuracy={(n_corrects/10000.0):0.6f}")
                # batch_size = 100
                
            # train the neural network
            first_fully_connected_layer.generate_dropout()
            input_layer.forward_propagation()

        end_time = time.perf_counter()
        print(f"This epoch took: {(end_time - start_time):0.6f}\n")
        print_current_metrics(epoch+1)

    return input_layer

def test_neural_network(test_images, test_labels, neural_network) -> None: 
    n_tests = len(test_labels)
    start_time = time.perf_counter()
    # for i in range(n_tests):
    prediction = neural_network.forward_propagation(test_images.T, False)
        # if prediction[0] == test_labels[i]: n_corrects += 1
    n_corrects = np.sum(prediction == test_labels)
    end_time = time.perf_counter()
    print(f"\nTesting accuracy = {(n_corrects/n_tests):0.6f}. This took {(end_time - start_time):0.6f} number of corrects={n_corrects}")


def main() -> None:
    """
    if (len(sys.argv) < 5):
        print("Too few args. Pass in ./program training_image_file training_label_file test_image_file test_label_file")
        return 
    """
    
    # training_image_path, training_label_path, epochs = sys.argv[1], sys.argv[2], 20
    epochs, n_neurons = 5, 100  
    
    start = time.perf_counter()
    # Read in training data
    print(f"Starting to read training and testing data for the neural network with {n_neurons} neurons")
    train_images, train_labels = read_training_data() 
    train_data = np.append(train_labels.reshape(1, len(train_labels)), train_images.T, axis = 0).T # each row is a training data

    test_images, test_labels = read_testing_data()
    # test_data = np.append(test_labels.reshape((1, len(test_labels))), test_images.T, axis = 0).T # each row is a testing data
    end = time.perf_counter()
    print(f"Finished reading all the images into the program, in {end - start} seconds")

    # Train the Neural Network
    print("Starting to train the neural network")
    neural_network = train_neural_network(train_images, train_labels, train_data, test_images, test_labels, epochs, n_neurons)
    print("Finish training the neural network")


    # Test the Neural Network
    print("\nStarting to test the neural network on testing data")
    test_neural_network(test_images, test_labels, neural_network)
    print("Finish testing the neural network")


if __name__ == "__main__":
    main()