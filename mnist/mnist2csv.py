#!/usr/bin/env python3
"""A tool to convert the MNIST database into human readable CSVs.

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

from csv import writer
from typing import List

def mnist2csv(inputs, labels, dataset, samples):
    images:List[List[int]] = []

    inputs.read(16) # discard magic number, number of images, number of rows and number of columns
    labels.read(8)  # discard magic number and number of items
    for _ in range(samples):
        image:List[int] = [0] * (28*28 + 1)
        for i in range(len(image) - 1):
            image[i] = ord(inputs.read(1))
        image[-1] = ord(labels.read(1))
        images.append(image)
    
    dataset.writerows(images)

def main():
    with open("train-images-idx3-ubyte", "rb") as inputs:
        with open("train-labels-idx1-ubyte", "rb") as labels:
            with open("source.csv", "w", newline='') as output:
                dataset = writer(output)
                mnist2csv(inputs, labels, dataset, 60000)
    with open("t10k-images-idx3-ubyte", "rb") as inputs:
        with open("t10k-labels-idx1-ubyte", "rb") as labels:
            with open("test.csv", "w", newline='') as output:
                dataset = writer(output)
                mnist2csv(inputs, labels, dataset, 10000)

if __name__ == '__main__':
    main()
