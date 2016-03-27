from scipy import stats
from numpy import unravel_index
from sklearn.cross_validation import KFold
import scipy as sp
import numpy as np
import random
import argparse
import csv

def commandLineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("hiddern_layers", help="Number of hidden nodes e.g. 2. If you want to add layers go to line 161 of the program and change args.hiddern_layers to 2,2,2", type=int)
    parser.add_argument("file", help="Input file name e.g. data.txt")
    args = parser.parse_args()
    return args


def read_data_file(file_name):
    data_file = []
    structure_file = []
    target_data = []
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data_file.append(list(row))
        random.shuffle(data_file)
        for row in data_file:
            target_data.append(row[-1])
            del row[-1]
    

    with open(file_name.replace(".txt", "") + '-structure.txt','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            structure_file.append(list(row))

    for i in range(len(data_file)):
        for j in range(len(data_file[i])):
            if data_file[i][j] == "?":
               data_file[i][j] = random.choice(structure_file[j + 1])

    return data_file, target_data, structure_file

def confident_interval_calculation(list_of_errors, confidence=0.95):
    a = 1.0*np.array(list_of_errors)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m-h, m+h


def error_calculation(validation_target, validation_output):
    count = 0
    for output_index in range(len(validation_output)):
        max_index = unravel_index(validation_output[output_index].argmax(), validation_output[output_index].shape)[0]
        if validation_target[output_index][max_index] == 1:
            count = count + 1

    return float(count)/len(validation_output)

def normalize_data(data_file, target_data, structure_file):
    normalized_target_data = []
    for i in range(len(data_file)):
        for j in range(len(data_file[i])):
            data_file[i][j] = float(structure_file[j + 1].index(data_file[i][j]) + 1)/len(structure_file[j+1])
    
    for i in range(len(target_data)):
        individual_target_data = [0]*len(structure_file[-1])
        individual_target_data[structure_file[-1].index(target_data[i])] = 1 
        normalized_target_data.append(individual_target_data)
    return data_file, normalized_target_data


def read_structure_file():
    structure_files = [];
    with open('structure_files.txt', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            structure_files.append(list(row))

    return structure_files


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


class BackPropagationNetwork:
    def __init__(self, layerSize):
        self.layerCount = len(layerSize) - 1  # layerSize is a tuple
        self.shape = layerSize  # e.g. (2,2,1)
        self._layerInput = []
        self._layerOutput = []
        self.shape = None
        self.weights = []
        self._previousWeightsDelta = []

        for (layer_one, layer_two) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(layer_two, layer_one + 1)))  # +1 for the bias node
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
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones(lnCases)]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(sigmoid(layerInput))

        return self._layerOutput[-1].T

    def train(self, input, target, trainingRate=0.0005, momentum=0):
        delta = []
        lnCases = input.shape[0]

        self.run(input)

        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                outputDelta = self._layerOutput[index] - target.T
                error = np.sum(outputDelta ** 2)
                delta.append(outputDelta * sigmoid_derivative(self._layerInput[index]))
            else:
                deltaBackward = self.weights[index + 1].T.dot(delta[-1])
                delta.append(deltaBackward[:-1, :] * sigmoid_derivative(self._layerInput[index]))

        for index in range(self.layerCount):
            deltaIndex = self.layerCount - 1 - index
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            currentWeightDelta = np.sum(
                layerOutput[None, :, :].transpose(2, 0, 1) * delta[deltaIndex][None, :, :].transpose(2, 1, 0)
                , axis=0)
            weightDelta = trainingRate * currentWeightDelta + momentum * self._previousWeightsDelta[index]
            self.weights[index] -= weightDelta
            self._previousWeightsDelta[index] = weightDelta
        return error


if __name__ == "__main__":
    # get command line arguments
    args = commandLineArguments()
    data_file, target_data, structure_file = read_data_file(args.file)
    normalized_data, normalized_target_data = normalize_data(data_file, target_data, structure_file)
    # inputLayers = (len(data_file[0]), args.hiddern_layers, len(structure_file[-1]))
    inputLayers = (len(data_file[0]), args.hiddern_layers, len(structure_file[-1]))

    dataInput = np.array(normalized_data) 
    target = np.array(normalized_target_data)#np.array([[0,1], [1,0], [1,0]])

    maxLoop = 10000
    minError = 1e-1

    kf = KFold(len(dataInput), n_folds=10)

    confidence_interval_errors = []
    iteration = 0
    
    for train, test in kf:
        iteration += 1
        bpn = BackPropagationNetwork(inputLayers)
        print ("Cross Validation:{0}".format(iteration))
        for i in range(maxLoop + 1):
            error = bpn.train(dataInput[train],target[train])
            if i % 1000 == 0:
                print("Iteration @ {0:d}K - Error: {1:0.4f}".format(int(i / 1000), error))
            if error <= minError:
                print("Termination at Iteration: {0}".format(i))
                break
        testing_output = bpn.run(np.array(dataInput[test]))         
        confidence_interval_errors.append(error_calculation(target[test], testing_output))
        
    print "C.I: {0}".format(confident_interval_calculation(confidence_interval_errors))
    print "mean: {0}".format(np.mean(confidence_interval_errors))
    print "std: {0}".format(np.std(confidence_interval_errors))



