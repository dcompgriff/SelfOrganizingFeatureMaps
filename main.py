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


def main():
    print("Running Main.")

    # Read data.
    data = pd.read_csv('./animals.csv')
    # Create SOFm grid.
    mSOFM = SOFM_Core.SOFMGrid(input_size=13 , row_size=10, col_size=10)
    # Train grid.
    mSOFM.train(data[data.columns[:-1]].values, epochs=1)
    # Show maximal activations.
    #tupList = mSOFM.getMaxActivations(data.values, classColumnIndex=data.shape[1] - 1)
    tupList = mSOFM.getAllNeuronActivations(data.values, classColumnIndex=data.shape[1] - 1)

    xVals = list(map(lambda item: item[0], tupList))
    yVals = list(map(lambda item: item[1], tupList))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(10, 8, forward=True)
    ax.scatter(xVals, yVals)
    for tup in tupList:
        ax.text(tup[0], tup[1], tup[2])
    plt.show()


if __name__ == "__main__":
    main()




