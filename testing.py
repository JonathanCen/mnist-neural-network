import numpy as np

def init_params():
  W1 = np.random.rand(10, 784)
  b1 = np.random.rand(10, 1)
  W2 = np.random.rand(10, 10)
  b2 = np.random.rand(10, 1) * b1
  return W1, b1, W2, b2


if __name__ == '__main__':
	W1, b1, W2, b2 = init_params()
	print('W1: ', W1)
	print('b1: ', b1)
	print('W2: ', W2)
	print('b2: ', b2)
