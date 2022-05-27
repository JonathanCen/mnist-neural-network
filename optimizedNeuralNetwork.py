from mlxtend.data import loadlocal_mnist
import platform
import sys
import numpy as np
import random
import time

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
        # start_time = time.perf_counter()
        # global train_images, train_labels
        if is_training:
            train_data = self.train_data[self.train_data_index : self.train_data_index + batch_size]
            train_images = np.array(list(map(lambda x : x[0], train_data))).reshape((batch_size, 784, 1))
            train_labels = np.array(list(map(lambda x : x[1], train_data)))
            self.train_data_index += batch_size
        # end_time = time.perf_counter()
        # print(f'Input layer took: {end_time - start_time}')
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
        self.weights = np.random.rand(n_neurons, n_incoming_neurons) - .5
        self.bias = np.random.rand(n_neurons, 1) - .5
        self.dropout = np.array(list(map(lambda x: 1/0.6 if x < 0.6 else 0, np.random.rand(n_neurons)))).reshape((n_neurons, 1))

        # ! possibly don't need this, because everytime we are going to just
        self.avg_weights_deriv = None
        self.avg_bias_deriv = None
        self.forward_pass_data = None

    def ReLU_activation_function(self, output):
        return np.maximum(0, output)

    # is_training - True, then modify the weights and bias
    #             - False, then return an array of predictions 
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        # start_time = time.perf_counter()
        self.forward_pass_data = forward_pass_data
        # print(self.weights.shape, forward_pass_data.shape, self.bias.shape)
        output = np.matmul(self.weights, forward_pass_data) + self.bias
        if not self.is_output_layer:
            output = self.ReLU_activation_function(output) * self.dropout
        # end_time = time.perf_counter()
        # print(output.shape)
        # print(f'FullyConnectedLayer {1 if self.n_neurons == 1024 else 2} took: {end_time - start_time}')
        predictions = self.next_layer.forward_propagation(output, is_training)
        return predictions

    def backward_propagation(self, backward_pass_data) -> None:
        global learning_rate

        downstream_deriv = np.matmul(self.weights.T, backward_pass_data)

        if self.is_output_layer:
            self.avg_weights_deriv = np.sum(np.matmul(backward_pass_data, self.forward_pass_data.reshape((batch_size, 1, self.n_incoming_neurons))), axis=0)
            self.avg_bias_deriv = np.sum(backward_pass_data, axis = 0)
        else:
            # downstream_deriv = np.matmul((self.dropout * backward_pass_data).T, self.weights)
            self.avg_weights_deriv = np.sum(np.matmul((self.dropout * backward_pass_data), self.forward_pass_data.reshape((batch_size, 1, self.n_incoming_neurons))), axis=0)
            self.avg_bias_deriv = np.sum(backward_pass_data * self.dropout, axis = 0)
        
        self.update_weights_and_bias(learning_rate)
        self.previous_layer.backward_propagation(downstream_deriv)


    def update_weights_and_bias(self, learning_rate) -> None:
        # print("updating weights and bias")
        # print(f"Avg weights deriv: {self.avg_weights_deriv}")
        # print(f"Avg bias deriv: {self.avg_bias_deriv}")
        self.weights -= learning_rate * self.avg_weights_deriv
        self.bias -= learning_rate * self.avg_bias_deriv
        self.avg_weights_deriv = None
        self.avg_bias_deriv = None


class SoftMaxLayer:
    def __init__(self, n_neurons, next_layer = None, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.next_layer = next_layer
        self.previous_layer = previous_layer
        self.forward_pass_output = None

    def extract_prediction(self):
        return self.forward_pass_output.argmax()
        # return self.forward_pass_output.argmax(axis = 1).reshape(batch_size)
    
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        """ # This is the formula I had when doing CUDA
        # self.forward_pass_output = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))
        # print(self.forward_pass_output)/
        """
        # start_time = time.perf_counter()
        # if batch_size == 1:
        #     print(forward_pass_data.shape)
        #     intermediate_output = np.exp(forward_pass_data - max(forward_pass_data))
        #     self.forward_pass_output = intermediate_output / sum(intermediate_output)
        # else:    

        intermediate_output = np.exp(forward_pass_data.reshape((batch_size, self.n_neurons)) - np.array(list(map(lambda x : max(x), forward_pass_data))))
        sum_intermediate_output = np.sum(intermediate_output, axis=1).reshape((batch_size, 1))
        self.forward_pass_output = (intermediate_output / sum_intermediate_output).reshape((batch_size, self.n_neurons, 1))
        print(intermediate_output.shape, self.forward_pass_output.shape)

        # end_time = time.perf_counter()
        # print(f'SoftMaxLayer took: {end_time - start_time}')

        return self.extract_prediction() if not is_training else self.next_layer.forward_propagation(self.forward_pass_output)

    def backward_propagation(self, backward_pass_data) -> None:
        # start_time = time.perf_counter()
        downstream_deriv = np.zeros((batch_size, self.n_neurons, 1), dtype=np.float64)

        #! try to optimize this if possible
        for train_index in range(batch_size):
            for neuron in range(self.n_neurons):
                accum = (backward_pass_data[train_index].T.dot(self.forward_pass_output[train_index])) * -self.forward_pass_output[train_index][neuron][0]

                # adjust for when neuron of SML = neuron of CEL
                accum -= (backward_pass_data[train_index][neuron][0] * self.forward_pass_output[train_index][neuron][0] * -self.forward_pass_output[train_index][neuron][0])
                accum += backward_pass_data[train_index][neuron][0] * self.forward_pass_output[train_index][neuron][0] * (1 - self.forward_pass_output[train_index][neuron][0])

                downstream_deriv[train_index][neuron][0] = accum

        # end_time = time.perf_counter()
        # print(f'SoftMaxLayer Back Prop took: {end_time - start_time}')
        self.previous_layer.backward_propagation(downstream_deriv)
        

class CrossEntropyLayer:
    def __init__(self, n_neurons, previous_layer = None) -> None:
        self.n_neurons = n_neurons
        self.previous_layer = previous_layer

    def forward_propagation(self, forward_pass_data) -> None:
        global train_labels
        # print(forward_pass_data.shape)
        downstream_deriv = np.zeros((batch_size, self.n_neurons, 1), dtype=np.float64)
        for index, label in enumerate(train_labels):
            if forward_pass_data[index][label][0] == 0:
                print(forward_pass_data[index])
                print("WTF is going on here")
            downstream_deriv[index][label][0] = -1/forward_pass_data[index][label][0]
        # print(train_labels[0])
        # print(downstream_deriv[0])
        self.previous_layer.backward_propagation(downstream_deriv)

def read_int(file_pointer: gzip.GzipFile) -> int:  
    """ Returns the int of the first 4 bytes
    """
    return int.from_bytes(file_pointer.read(4), "big")

def normalize_images(images) -> np.ndarray:
    normalizerLambda = lambda x: (x/127.5) - 1
    return np.array(list(map(normalizerLambda, images)))

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

    return input_layer





def main() -> None:
    """
    if (len(sys.argv) < 5):
        print("Too few args. Pass in ./program training_image_file training_label_file test_image_file test_label_file")
        return 
    """
    
    # training_image_path, training_label_path, epochs = sys.argv[1], sys.argv[2], 20
    epochs, n_neurons = 20, 128
    
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
    neural_network = train_neural_network(train_images, train_labels, train_data, n_neurons, epochs)
    print("Finish training the neural network")

    """
    # Test the Neural Network
    print("\nStarting to test the neural network on testing data")
    test_neural_network(testing_images, testing_labels, neural_network)
    print("Finish testing the neural network")
    """

if __name__ == "__main__":
    main()