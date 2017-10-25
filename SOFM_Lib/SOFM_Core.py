'''
Author: Daniel Griffin
Date: 4/15/2016
Description: This represents the core code for building and
training SOFM nets. The code was made into a module initially
so that it would be easier to incorporate it as an actual
library, should the need or desire ever arise.
'''

import math
import numpy as np

class SOFMGrid:

    def __init__(self, input_size=1, row_size=1, col_size=1):
        # Eta, learning rate. Initially, set as constant.
        self.eta0 = .1
        self.eta_tau = 2000
        # Time constant for the decaying sigma.
        self.sigma0 = 5
        self.sigma_tau = 1243
        #self.sigma_tau = 50
        # SOFM Grid size.
        self.row_size = row_size
        self.col_size = col_size
        self.neurons = []
        self.input_size = input_size
        # Make a list of neuorns, where the positions are set based on the grid and row size.
        for row in range(0, row_size):
            for col in range(0, col_size):
                self.neurons.append(Neuron(row, col, input_size=self.input_size, weight_lbound=-0.05, weight_ubound=0.05))

    '''
    This function expects a 2 dimensional numpy array, with
    each row representing a data example.

    Training Steps:
     1) Select a data point xj.
     2) Find position of max neuron, i* = argmin_i(||xj - wi||)
     3) Update the weights in the network over all i as dwij = n*N(i,i*,t)*(xj - wij)
        N(i, i*, t) = exp(-1*( ( -(||ri - ri*||)^2 )/(2sigma(t)^2) )
            -ri is the position of neuron i in the grid.
            -ri* is the position of the winning neuron i* in the grid.
        sigma(t) = sigmo_0*exp(-t/tau)
            -Tau is a time constant (Typically around 1000ish).
    '''
    def train(self, train_data, organize_epochs=2000, finetune_epochs=10000):
        # Time variable for the decaying sigma.
        time = 0
        # Get a copy of the training data that can be shuffled.
        mTrainData = train_data.copy()

        # Organizing phase.
        for epoch in range(0, organize_epochs):
            print("Epoch: " + str(epoch))
            # Shuffle the training data.
            #np.random.shuffle(mTrainData)
            # Loop over each datapoint.
            for dataPoint in mTrainData[:]:
                # Get the row and column for the closest neuron.
                row, col, index = self.findMaxActivatingNeuron(dataPoint)
                # Update all weights.
                self.updateWeights(index, dataPoint, time)
                #Increment the time parameter.
                time += 1
                
        # Fine tuning phase.
        self.eta0 = 0.001
        self.sigma0 = 1
        time = 0
        for epoch in range(0, finetune_epochs):
            print("Fine tune epoch: " + str(epoch))
            # Shuffle the training data.
            np.random.shuffle(mTrainData)
            # Loop over each datapoint.
            for dataPoint in mTrainData[:]:
                # Get the row and column for the closest neuron.
                row, col, index = self.findMaxActivatingNeuron(dataPoint)
                # Update all weights.
                self.updateWeights(index, dataPoint, time)

    '''
    Update the weights in the network using a radial distance function.
    '''
    def updateWeights(self, maxNeuronPos, x, t):
        for neuronIndex in range(0, len(self.neurons)):
            neuron = self.neurons[neuronIndex]
            for i in range(0, neuron.weights.shape[0]):
                # Calculate delta wi using the time decaying radial distance function.
                dwi = (self.eta(t) * self.radialDist(neuronIndex, maxNeuronPos, t) * (x[i] - neuron.weights[i]))
                # Update the weights.
                neuron.weights[i] = neuron.weights[i] + dwi

    '''
    Calculate the radial distance function.
    '''
    def radialDist(self, i, i_max, t):
        # Form the positions into vectors since they are stored as individual values.
        i_pos = np.array([self.neurons[i].row, self.neurons[i].col])
        i_max_pos = np.array([self.neurons[i_max].row, self.neurons[i_max].col])
        # Calculate the time-shrinking, radial distance function.
        radDist = math.exp( -1.0*(math.pow(np.linalg.norm(i_pos - i_max_pos),2)) / (2.0*math.pow(self.sigma(t),2)) )
        return radDist

    '''
    Calculate the time decaying sigma value.
    '''
    def sigma(self, t):
        return float(self.sigma0) * math.exp(float(-t)/self.sigma_tau)

    '''
    Calculate the time decaying eta (learning rate) value.
    '''
    def eta(self, t):
        return float(self.eta0)*math.exp(float(-t)/self.eta_tau)

    '''
    Find the neuron whose weight vector is closest to the input vector x.
    '''
    def findClosestNeuron(self, x):
        maxPos = (0, 0)
        minDist = 1000000000
        maxIndex = 0
        for i in range(0, len(self.neurons)):
            dist = np.linalg.norm(self.neurons[i].weights - x)
            if dist < minDist:
                minDist = dist
                maxPos = (self.neurons[i].row, self.neurons[i].col)
                maxIndex = i

        return maxPos[0], maxPos[1], maxIndex
        
    def findMaxActivatingNeuron(self, x):
        maxIndex = 0
        maxOutput = 0
        pos = (0, 0)
        for i in range(0, len(self.neurons)):
            output = self.neurons[i].output(x)
            if output > maxOutput:
                maxOutput = output
                maxIndex = i
                pos = (self.neurons[i].row, self.neurons[i].col)
                
        return pos[0], pos[1], maxIndex
            

    '''
    
    '''
    def getGridResponses(self, x):
        responseTupleList = []
        for neuron in self.neurons:
            responseTupleList.append((neuron.row, neuron.col, neuron.output(x)))
        
        return responseTupleList

    '''
    Returns a list of tuples, where each tuple is a neuron with:
    (<row>, <col>, <class>).
    '''
    def getMaxActivations(self, data, classLabels):
        # Find Max Activations and label them.
        tupList = []
        for index in range(0, data.shape[0]):
            row, col, i = self.findMaxActivatingNeuron(data[index])
            tupList.append((row, col, classLabels[index]))

        return tupList

    '''
    Get the grid of all neurons, and which animal class they are closest to.
    '''
    def getAllNeuronActivations(self, data, classLabels):
        # Find data point for which each neuron is closest to.
        tupList = []
        for neuron in self.neurons:
            maxOutput = 0
            closestLabel = None
            # Loop through all animal points, and find the closest one.
            for index in range(0, data.shape[0]):
                output = neuron.output(data[index])
                if output > maxOutput:
                    closestLabel = classLabels[index]
                    maxOutput = output
            # Append the closest animal label tuple to the tupList.
            tupList.append((neuron.row, neuron.col, closestLabel))

        return tupList

    '''
    Return x, y of time response of the sigma function over time.
    '''
    def plotSigmaOverTime(self, time_steps=1000):
        x = np.arange(0, time_steps, 1)
        y = np.zeros(time_steps)
        for step in range(0, time_steps):
            y[step] = self.sigma0 * math.exp(-float(step)/self.sigma_tau)

        return x, y

        '''
    Return x, y of time response of the eta function over time.
    '''
    def plotEtaOverTime(self, time_steps=1000):
        x = np.arange(0, time_steps, 1)
        y = np.zeros(time_steps)
        for step in range(0, time_steps):
            y[step] = self.eta0 * math.exp(-float(step)/self.eta_tau)

        return x, y

'''
SOFM neuron abstraction.
'''
class Neuron:
    def __init__(self, row, col, input_size=1, weight_lbound = -0.05, weight_ubound = 0.05):
        self.weights = np.random.uniform(weight_lbound, weight_ubound, input_size)
        self.row = row
        self.col = col

    '''
    Return the distance, or magnitude between a point x and this neuron's weight vector.
    '''
    def dist(self, x):
        return scipy_dist.euclidean(x, self.weights)
    '''
    Return the dot product of the neuron weights and the input data.
    '''
    def output(self, x):
        return np.dot(self.weights, x)









