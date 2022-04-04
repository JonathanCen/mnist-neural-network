
import sys
import numpy as np

import gzip

import matplotlib.pyplot as plt

"""
Neural Network Design:

Input layer: 
"""

class Neuron:
    def __init__(self) -> None:
        pass




def read_int(file_pointer: gzip.GzipFile) -> int:  
    """ Returns the int of the first 4 bytes
    """
    return int.from_bytes(file_pointer.read(4), "big")

def read_training_data(training_image_path: str, training_label_path: str) -> None:
    """ Returns a numpy array of the training images and labels

    """

    # Process the training images
    with gzip.open(training_image_path, 'r') as reader:
        # Read the meta data 
        reader.read(4) # Read past the magic number
        n_images, n_rows, n_cols = read_int(reader), read_int(reader), read_int(reader)

        training_images = np.empty((n_rows, n_cols, n_images), dtype=np.float16)

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




def train_neural_network(training_images, training_labels) -> None:
    pass


def main() -> None:
    """
    if (len(sys.argv) < 5):
        print("Too few args. Pass in ./program training_image_file training_label_file test_image_file test_label_file")
        return 
    """
    
    training_image_path, training_label_path = sys.argv[1], sys.argv[2]

    # Read in training data
    training_images, training_labels = read_training_data(training_image_path, training_label_path)
    print("Finished reading all the images into the program.")

    # Train the Neural Network
    train_neural_network(training_images, training_labels)

    # Test the Neural Network

if __name__ == "__main__":
    main()