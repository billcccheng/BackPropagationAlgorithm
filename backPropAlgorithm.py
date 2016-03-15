import numpy as np
import argparse

def commandLineArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("layer", help="e.g. 2,2,2,1 means 2 input nodes, layer 1 hidden with 2 nodes, layer 2 hidden with 2 nodes, 1 output node")
	
	args = parser.parse_args()
	arg_arrays = (args.layer.split(","))
	arg_arrays = [int(x) for x in arg_arrays]
	return tuple(arg_arrays)

def readDataFile:
	data_files = [];
	with open('data.txt', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data_files.append(list(row))
    return data_files

def readStructureFile:
	structure_files = [];
	with open('data.txt', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        structure_files.append(list(row))
    return structure_files


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
		self._previousWeightsDelta = []

		for(layer_one, layer_two) in zip(layerSize[:-1], layerSize[1:]):
			self.weights.append(np.random.normal(scale = 0.1, size = (layer_two, layer_one + 1))) # +1 for the bias node
			self._previousWeightsDelta.append(np.zeros((layer_two, layer_one + 1)))

	def run(self, input):
		lnCases = input.shape[0]
		self._layerInput = []
		self._layerOutput = []
		for index in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones(lnCases)]))
				# print self.weights[0], np.vstack([input.T, np.ones([1, lnCases])])
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones(lnCases)]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(sigmoid(layerInput))

		return  self._layerOutput[-1].T

	def train(self, input, target, trainingRate = 0.3, momentum = 0.9):
		delta = []
		lnCases = input.shape[0]

		self.run(input)

		for index in reversed(range(self.layerCount)):
			if index == self.layerCount - 1:
				outputDelta = self._layerOutput[index] - target.T
				error = np.sum(outputDelta**2)
				delta.append(outputDelta * sigmoid_derivative(self._layerInput[index]))
			else:
				deltaBackward = self.weights[index + 1].T.dot(delta[-1])
				delta.append(deltaBackward[:-1,:] * sigmoid_derivative(self._layerInput[index]))
		
		for index in range(self.layerCount):
			deltaIndex = self.layerCount - 1 - index	
			if index == 0:
				layerOutput = np.vstack([input.T, np.ones([1,lnCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])	

			currentWeightDelta = np.sum(
                                layerOutput[None,:,:].transpose(2, 0 ,1) * delta[deltaIndex][None,:,:].transpose(2, 1, 0)
                                , axis = 0)
 
			weightDelta = trainingRate * currentWeightDelta + momentum * self._previousWeightsDelta[index]
			self.weights[index] -= weightDelta
			self._previousWeightsDelta[index] = weightDelta		
		return error



if __name__ == '__main__':
    layers = commandLineArguments()
    bpn = BackPropagationNetwork(layers)

    dataInput = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    target = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
    
    
    maxLoop = 1000000
    minError = 1e-3
    for i in range (maxLoop +1 ):
    	error = bpn.train(dataInput, target)
    	if i % 10000 == 0:
    		print("Iteration {0:d}K - Error: {1:0.4f}".format(int(i/1000), err))
    	if error <= minError:
    		print("Termination at Iter: {0}".format(i))
    		break


    output = bpn.run(dataInput)
    print ("Input:\n{0}\n Output:\n {1}".format(dataInput, output))