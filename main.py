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
from SOFM_Lib import SOFM_Core
import matplotlib.pyplot as plt

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
    mSOFM.train(dataValues, epochs=10)
    # Show maximal activations.
    #tupList = mSOFM.getMaxActivations(data.values, classColumnIndex=data.shape[1] - 1)
    #tupList = mSOFM.getAllNeuronActivations(dataValues, classColumnIndex=data.shape[1] - 1)

    exWeights = mSOFM.neurons[0].weights
    print("Example Weights: ")
    print(exWeights)

    # xVals = list(map(lambda item: item[0], tupList))
    # yVals = list(map(lambda item: item[1], tupList))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # fig.set_size_inches(10, 8, forward=True)
    # ax.scatter(xVals, yVals)
    # for tup in tupList:
    #     ax.text(tup[0], tup[1], tup[2])
    # plt.show()


if __name__ == "__main__":
    main()




