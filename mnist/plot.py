#!/usr/bin/env python3
"""A tool to plot samples from the MNIST database in CSV format.

This file is part of a hack distributed under the Hacking License (see HACK.txt)

Copyright (C) 2021 Giacomo Tesio
"""

__author__ = "Giacomo Tesio"
__contact__ = "giacomo@tesio.it"
__copyright__ = "Copyright 2021, Giacomo Tesio"
__date__ = "2021/09/01"
__deprecated__ = False
__email__ =  "giacomo@tesio.it"
__license__ = "Hacking License"
__maintainer__ = "Giacomo Tesio"
__status__ = "Proof of Concept"
__version__ = "1.0.0"

import sys
from csv import reader
import matplotlib.pyplot as plt

def plotSample(csvfile:str, index:int) -> None:
    rowNum = 0
    with open(csvfile, 'r') as file:
        csv = reader(file)
        for row in csv:
            if rowNum == index:
                line = [int(x.strip()) for x in row[:-1]]
                img = [list(line[i:i+28]) for i in range(0, len(line), 28)]
                plt.imshow(img, interpolation='nearest', cmap='Greys')
                print(row[-1])
                plt.show()
                return
            rowNum += 1

def help() -> None:
    hs = """
plot.py n set.csv

Plot the sample at row n from set.csv
"""
    print(hs)
    sys.exit()

def main(argv:list):
    if len(argv) != 3:
        help()
    plotSample(argv[2], int(argv[1]))
    
if __name__ == '__main__':
    main(sys.argv)
