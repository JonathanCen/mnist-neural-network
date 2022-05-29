import time
import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import OneHotEncoder

""" --- Global Variables --- """
learning_rate = 0.001   # learning_rate - the rate which parameters change by on each update
batch_size = 100        # batch_size - the number of training images before NN parameters are updated
train_images, train_labels = None, None

## Times how long each operation takes
# accum_time_FCL1_forward_prop = 0
# accum_time_FCL2_forward_prop = 0
# accum_time_FCL1_backward_prop = 0
# accum_time_FCL2_backward_prop = 0
# accum_time_SML_forward_prop = 0
# accum_time_SML_backward_prop = 0
# accum_time_CEL_forward_prop = 0


""" --- Layers of the Neural Network --- """

""" InputLayer """
class InputLayer:
    def __init__(self, train_data, next_layer = None) -> None:
        # Connect the layers
        self.next_layer = next_layer

        # Training data
        self.train_data = np.random.permutation(train_data)
        self.train_data_index = 0 

    def shuffle_training_data(self) -> None:
        """ Shuffle the the training data """

        np.random.shuffle(self.train_data)
        self.train_data_index = 0

    def generate_training_data(self) -> None:
        global train_images, train_labels
        train_data = self.train_data[self.train_data_index : self.train_data_index + batch_size].T
        train_images, train_labels = train_data[1:], train_data[0].astype(np.int32)
    
    def forward_propagation(self, test_images = None, is_training = True) -> [int]:
        """ Returns the prediction[s] of the input or trains the model 

        If test_images is None, then it trains the model
        Otherwise, it returns prediction[s] 
        """

        # If the model is training, then generate training data, and incrememnt the index
        if is_training:
            self.generate_training_data()
            self.train_data_index += batch_size
            return self.next_layer.forward_propagation(train_images, is_training)
        
        # If the model is testing, then simply call the next layer
        return self.next_layer.forward_propagation(test_images, is_training)

    def back_propagation(self, backward_pass_data) -> [int]:
        """ Simply returns [-1] to the previous layer """ 
        return [-1]


""" FullyConnectedLayer """
class FullyConnectedLayer:
    def __init__(self, n_neurons, n_incoming_neurons, is_output_layer=False, next_layer = None, previous_layer = None) -> None:
        # Connect the layers
        self.next_layer = next_layer
        self.previous_layer = previous_layer

        # Data about the layer
        self.n_neurons = n_neurons
        self.is_output_layer = is_output_layer
        self.n_incoming_neurons = n_incoming_neurons
        self.forward_pass_data = None

        # Initialize weights, bias, and dropout (forward prop)
        self.weights = np.random.rand(n_neurons, n_incoming_neurons) - .5 # (n, 784) or (10. n)
        self.bias = np.random.rand(n_neurons, 1) - .5
        self.dropout = None

    def generate_dropout(self):
        """ Randomly generates the dropout of this layer"""
        droupoutRateLambda = lambda x: 1/0.6 if x < 0.6 else 0
        self.dropout = np.array(list(map(droupoutRateLambda, np.random.rand(self.n_neurons)))).reshape((self.n_neurons,1))

    def ReLU_activation_function(self, output):
        """ Passes the input through ReLU activation function """
        return np.maximum(0, output)

    def ReLU_gradient(self, upstream_gradient):
        """ Computes the ReLU derivative/gradient 
        
        Returns a np.ndarray of bools, where 
            True is neurons that didn't ouput 0 from ReLU
            False is neurons that did ouptut 0 from ReLU 
        """
        return upstream_gradient > 0

    def forward_propagation(self, forward_pass_data, is_training) -> [int]:
        """ For each neuron compute an output by input data * weights

        Takes forward_pass_data (np.array) and is_training (bool).
        If is_training is True, then modify the weights and bias and return [-1]
                       is False, then return an array of predictions 
        """

        # global accum_time_FCL1_forward_prop 
        # global accum_time_FCL2_forward_prop
        # start_time = time.perf_counter()

        # Store the input data, for computing the backpropagation
        self.forward_pass_data = forward_pass_data
        
        # Matrix multiplication to compute the sum of all multiplication of weights connecting to each neuron and input data
        output = (self.weights @ forward_pass_data) + self.bias 

        # If it's not the output layer, then pass the output through the ReLU activation function
        if self.is_output_layer == False:
            output = self.ReLU_activation_function(output) 

            # If it's training, then allow the NN to generalize better dropping out some neurons
            if is_training == True: output *= self.dropout

        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # if is_training:
        #     if self.is_output_layer: accum_time_FCL2_forward_prop += total_time
        #     else: accum_time_FCL1_forward_prop += total_time
        
        return self.next_layer.forward_propagation(output, is_training)

    def back_propagation(self, upstream_gradient) -> None:
        """ Computes the upstream gradient for each training dataset and updates weights and biases """

        # global accum_time_FCL1_backward_prop, accum_time_FCL2_backward_prop
        # start_time = time.perf_counter()

        # If the layer is not output layer, then the upstream_gradient needs to account for the dropouts in the neural network 
        if self.is_output_layer == False: upstream_gradient = self.dropout * upstream_gradient

        # Downstream gradient is the change of the loss with respect to each weight 
        downstream_gradient = self.weights.T @ upstream_gradient
        if self.is_output_layer == False: downstream_gradient *= self.ReLU_gradient(self.forward_pass_data)

        # Weights derivate is dL/dw = dL/df * df/dw = upstream gradient * forward pass data, since df/dw = forward pass data
        avg_weights_deriv = (upstream_gradient @ self.forward_pass_data.T) / batch_size

        # Bias derivate is dL/db = dL/df * df/db = upstream_gradient, since df/db = 1 (add gate)
        avg_bias_deriv = np.sum(upstream_gradient, axis = 1).reshape(self.n_neurons, 1) / batch_size

        """
        if self.is_output_layer:
            downstream_gradient = self.weights.T @ upstream_gradient
            avg_weights_deriv = (upstream_gradient @ self.forward_pass_data.T) / batch_size
            avg_bias_deriv = np.sum(upstream_gradient, axis = 1).reshape(self.n_neurons, 1) / batch_size
        else:
            downstream_gradient = self.weights.T @ (self.dropout * upstream_gradient) * self.ReLU_gradient(self.forward_pass_data)
            avg_weights_deriv = ((self.dropout * upstream_gradient) @ self.forward_pass_data.T) / batch_size
            avg_bias_deriv = np.sum(self.dropout * upstream_gradient, axis = 1).reshape(self.n_neurons, 1) / batch_size
        """
        
        # Update the weights and bias 
        self.update_weights_and_bias(avg_weights_deriv, avg_bias_deriv)
        self.weights -= learning_rate * avg_weights_deriv
        self.bias -= learning_rate * avg_bias_deriv

        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # if self.is_output_layer: accum_time_FCL2_backward_prop += total_time
        # else: accum_time_FCL1_backward_prop += total_time

        self.previous_layer.back_propagation(downstream_gradient)

    def update_weights_and_bias(self, avg_weights_deriv, avg_bias_deriv) -> None:
        """ Updates the weights and bias of the neurons """
        self.weights -= learning_rate * avg_weights_deriv
        self.bias -= learning_rate * avg_bias_deriv


""" SoftMaxLayer """
class SoftMaxLayer:
    def __init__(self, n_neurons, next_layer = None, previous_layer = None) -> None:
        # Connect the layers
        self.next_layer = next_layer
        self.previous_layer = previous_layer

        # Data about the layer
        self.n_neurons = n_neurons
        self.softmax = None 

    def extract_prediction(self):
        """ Extracts the prediction for each training entry """
        return self.softmax.argmax(axis = 0)
    
    def forward_propagation(self, forward_pass_data, is_training) -> None:
        """ Computes the softmax for each training data.
        
        The formula to compute the softmax for a training data:
            softmax_i = exp(forward_pass_i) / sum(exp(forward_pass)),
            where softmax_i is the softmax for ith neuron, and 
                  forward_pass_i is the previous layer output for the ith neuron
        """

        # global accum_time_SML_forward_prop
        # start_time = time.perf_counter()

        self.softmax = np.exp(forward_pass_data) / sum(np.exp(forward_pass_data))

        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # if is_training: accum_time_SML_forward_prop += total_time

        return self.next_layer.forward_propagation(self.softmax) if is_training else self.extract_prediction()

    def back_propagation(self) -> None:
        """ Computes the upstream gradient for the next layer.

        SoftMaxLayer + CrossEntropyLayer upstream gradient equation:
            dL/dn_i (change of the loss with respect to each neuron in this layer) 
                    = softmax_i - actual_i,
            where softmax_i is the softmax output of neuron i and 
            actual_i is the actual output from the training label
        """

        # global accum_time_SML_backward_prop
        # start_time = time.perf_counter()

        # Since, there is only one correct label, we only subtract 1 from the neuron that represents the correct label
        # otherwise, we will subtract 0 from the neuron that represents the incorrect label = not changing any of the values
        self.softmax[train_labels, np.arange(batch_size)] = self.softmax[train_labels, np.arange(batch_size)] - 1
        
        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # accum_time_SML_backward_prop += total_time
        self.previous_layer.back_propagation(self.softmax)
        

""" CrossEntropyLayer """
class CrossEntropyLayer:
    def __init__(self, n_neurons, previous_layer = None) -> None:
        # Connect the layers
        self.previous_layer = previous_layer

        # Data about the layer
        self.n_neurons = n_neurons

    def forward_propagation(self, softmax) -> None:
        """ Computes the loss for all training entries

        Takes in a softmax from the softmax layer of shape (10, batch_size),
        and computes the loss for each training entry. Currently not using this method and computing
        the gradient with the softmax graident in SoftMax.backpropagation.
        """

        # global accum_time_CEL_forward_prop
        # start_time = time.perf_counter()

        # Computes the loss for each training entry, but it's not needed as it gets cancelled out when doing the gradient of softmax
        # loss = -np.log(softmax[train_labels, np.arange(len(train_labels))]) 
        
        # end_time = time.perf_counter()
        # total_time = end_time - start_time
        # accum_time_CEL_forward_prop += total_time

        self.previous_layer.back_propagation()


""" --- Reading in training and testing images --- """

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
    """ Prints the average time each part of the neural network for each epoch.
    """
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
    """ Connects different parts of the neural network. 
    
    Takes in a list of the different parts of the neural network and modifies the 
    next_layer and previous_layer attributes of each layer.
    """
    for i in range(len(list_layers)):
        if i - 1 > -1: list_layers[i].previous_layer = list_layers[i-1]
        if i + 1 < len(list_layers): list_layers[i].next_layer = list_layers[i+1]

def generate_random_test(train_data, num_tests):
    permutated_train_data = np.random.permutation(train_data[:num_tests]).T
    test_images, test_labels = permutated_train_data[1:], permutated_train_data[0].astype(np.int32)
    return test_images, test_labels

def train_neural_network(train_data, n_neurons, epochs, batch_size, learning_rate=0.001) -> None:
    """ Constructs and trains the neural network using mini-batch gradient descent.

    This function takes train_data (np.ndarray) for the neural network to train on, and parameters for the network:
    n_neurons (int) the number of neurons in the network, epochs (int) the number of times it passes through the training dataset,
    batch_size (int) the number of training entries before performing gradient descent, and learning_rate (float) the amount that
    the weights of the neural network is updated during training.
    """

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

        # Shuffle the training images and labels 
        input_layer.shuffle_training_data()

        print(f"\n----------------------- STARTING EPOCH {epoch} -------------------\n")
        start_time = time.perf_counter()

        for c_round in range(600):
            # Test the neural network every 100 rounds for each epoch
            if c_round % 100 == 0:
                validation_images, validation_labels = generate_random_test(train_data, 10000)
                prediction = input_layer.forward_propagation(validation_images, False)
                n_corrects = np.sum(prediction == validation_labels)
                print(f"Epoch {epoch}: Round {c_round:5d}: accuracy={(n_corrects/10000.0):0.6f} with {n_corrects} corrects")
                
            # Train the neural network
            first_fully_connected_layer.generate_dropout()
            input_layer.forward_propagation()

        end_time = time.perf_counter()
        print(f"This epoch took: {(end_time - start_time):0.6f}\n")
        # print_current_metrics(epoch+1)

    return input_layer

def test_neural_network(test_images, test_labels, neural_network) -> None: 
    """ Tests the neural network based on the test images, and test labels. 
    
    Takes test_images (np.ndarray), test_labels (np.ndarray), and neural_network (only the input layer).
    Prints out the percentage of accuracy of the model and the time it took to run all the test data.
    """
    n_tests = len(test_labels)
    start_time = time.perf_counter()

    prediction = neural_network.forward_propagation(test_images.T, False)
    n_corrects = np.sum(prediction == test_labels)

    end_time = time.perf_counter()
    print(f"\nTesting accuracy = {(n_corrects/n_tests):0.6f} with {n_corrects} corrects. This took {(end_time - start_time):0.6f} seconds")


def main() -> None:
    """
    if (len(sys.argv) < 5):
        print("Too few args. Pass in ./program training_image_file training_label_file test_image_file test_label_file")
        return 
    """

    # Read in training and testing data
    start = time.perf_counter()
    print(f"Starting to read training and testing data for the neural network")
    train_images, train_labels = read_training_data() 
    train_data = np.append(train_labels.reshape(1, len(train_labels)), train_images.T, axis = 0).T # each row is a training data
    test_images, test_labels = read_testing_data()
    # test_data = np.append(test_labels.reshape((1, len(test_labels))), test_images.T, axis = 0).T # each row is a testing data
    end = time.perf_counter()
    print(f"Finished reading all the images into the program, in {end - start} seconds")

    # Parameters for the Neural Network
    batch_size, epochs, n_neurons = 100, 5, 100

    # Train the Neural Network
    print(f"Starting to train the neural network with {n_neurons} neurons")
    neural_network = train_neural_network(train_data, n_neurons, epochs, batch_size)
    print("Finish training the neural network")


    # Test the Neural Network
    print("\nStarting to test the neural network on testing data")
    test_neural_network(test_images, test_labels, neural_network)
    print("Finish testing the neural network")


if __name__ == "__main__":
    main()