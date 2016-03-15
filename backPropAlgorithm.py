import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))


class BackPropagationNetwork:
	layerCount = 0
	shape = None
	weights = []

	def __init__(self, layerSize):
		self.layerCount = len(layerSize) - 1 #layerSize is a tuple
		self.shape = layerSize # e.g. (2,2,1)

		self._layerInput = []
		self._layerOutput = []


		for(input_layer, hidden_layer) in zip(layerSize[:-1], layerSize[1:]):
			self.weights.append(np.random.normal(scale = 0.1, size = (hidden_layer, input_layer+1)))


if __name__ == '__main__':
    bpn = BackPropagationNetwork((2,2,1)) # fetch argument in command line