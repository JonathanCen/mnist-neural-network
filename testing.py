import numpy as np

def init_params():
  W1 = np.random.rand(10, 784)
  b1 = np.random.rand(10, 1)
  W2 = np.random.rand(10, 10)
  b2 = np.random.rand(10, 1) * b1
  return W1, b1, W2, b2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    print('one_hot_Y: ', one_hot_Y)
    one_hot_Y[np.arange(Y.size), Y] = 1
    print('one_hot_Y: ', one_hot_Y)
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def dotProduct(X, Y):
	return X.dot(Y)

if __name__ == '__main__':
	W1, b1, W2, b2 = init_params()
	#print('W1: ', W1)
	#print('b1: ', b1)
	#print('W2: ', W2)
	#print('b2: ', b2)
	Y = np.random.randint(3, size=3)
	print('Y: ', Y)
	print('return: ', one_hot(Y))
	print('Dot product result: ', dotProduct(W2, b2))
	res = dotProduct(b2.T, W2)
	print('Dot product result2: ', res)
	res += res	
	print('Dot product res: ', res)
