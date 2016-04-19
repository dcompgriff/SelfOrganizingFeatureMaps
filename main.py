'''
Author: Daniel Griffin
Date: 4/15/2016
Description: This module is used to implement answers to the questions given in
the intelligent systems class at the university of cincinnati on self organizing
feature maps. It uses a "Library" that has all of the code related to building and
training the SOFM core.
'''

import numpy as np
import pandas as pd
import scipy as scipy
import pickle
from SOFM_Lib import SOFM_Core
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def getOneHotDict(nameSet, scale):
    mDict = {}
    currentIndex = 0
    for name in nameSet:
        oneHotArray = np.zeros((len(nameSet),))
        oneHotArray[currentIndex] = 1*scale
        mDict[name] = oneHotArray
        currentIndex += 1

    return mDict

def main():
    global mSOFM
    print("Running Main.")

    # Read data.
    data = pd.read_csv('./animals.csv')

    # Make a 'one-hot' encoding of each animal class label.
    oneHotDict = getOneHotDict(data['name'], 0.2)
    oneHotLabelsList = []
    for index, row in data.iterrows():
        oneHotLabelsList.append(oneHotDict[row['name']])

    # Parse data, scale it, and rebuild it so that the bit vectors are normalized.
    dataToScale = data[data.columns[:-1]].values
    animals = data[data.columns[-1]].values
    i = 0
    dataToScale = dataToScale.astype(np.float32)
    for row in dataToScale[:]:
        mean = row.mean()
        dataToScale[i] = dataToScale[i] * mean
        i += 1
    data = pd.DataFrame(dataToScale, columns=data.columns[:-1])
    data['name'] = animals

    # Create SOFm grid.
    mSOFM = SOFM_Core.SOFMGrid(input_size=29 , row_size=10, col_size=10)
    # Train grid.
    dataValues = data[data.columns[:-1]].values
    dataValues = np.hstack((dataValues, np.array(oneHotLabelsList)))
    mSOFM.train(dataValues, organize_epochs=2000, finetune_epochs=50000)

    # Create a data set for probing, where each animal label has attributes 0.
    zeroAttrData = np.zeros((data[data.columns[:-1]].values.shape))
    zeroAttrData = np.hstack((zeroAttrData, np.array(oneHotLabelsList)))
    tupList = mSOFM.getMaxActivations(zeroAttrData, data['name'])
    tups = list(map(lambda item: (item[0], item[1]), tupList))
    for row in range(0, 10):
        for col in range(0, 10):
            if (row, col) not in tups:
                tupList.append((row, col, '-'))

    exWeights = mSOFM.neurons[0].weights
    print("Example Weights: ")
    print(exWeights)

    xVals = list(map(lambda item: item[0], tupList))
    yVals = list(map(lambda item: item[1], tupList))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(10, 8, forward=True)
    ax.scatter(xVals, yVals)
    for tup in tupList:
        ax.text(tup[0], tup[1], tup[2])
    plt.show()

    # Find the closest class for each neuron.
    tupList = mSOFM.getAllNeuronActivations(zeroAttrData, classLabels=data['name'])

    xVals = list(map(lambda item: item[0], tupList))
    yVals = list(map(lambda item: item[1], tupList))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(10, 8, forward=True)
    ax.scatter(xVals, yVals)
    for tup in tupList:
        ax.text(tup[0], tup[1], tup[2])
    plt.show()
    
    tupList = mSOFM.getGridResponses(dataValues[0])
    xVals = list(map(lambda item: item[0], tupList))
    yVals = list(map(lambda item: item[1], tupList))
    zVals = list(map(lambda item: item[2], tupList))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xVals, yVals, zVals)
    plt.show()
    
    # Save the SOFM in a file.
    with open('./mSOFM.pkl', 'wb') as f:
        pickle.dump(mSOFM, f)

if __name__ == "__main__":
    main()




