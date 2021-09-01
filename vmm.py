"""Fundamental components of a Vector Mapping Machine (aka "artificial neural network").

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

from math import exp
from dataclasses import dataclass
from typing import List

@dataclass
class VectorMapper:
    """
    A vector mapper maps a vector from a space of inputSize
    dimensions to a vector from a space of outputSize dimensions.
    """
    inputSize:int
    outputSize:int
    def map(self, inputs:List[float]) -> List[float]:
        pass

@dataclass
class VectorReducer(VectorMapper):
    """
    An artificial neuron in AI/ML parlance.

    It's a vector mapper that can fold a vector of N-dimension
    into a singleton (a one-dimensional vector).

    It's the fundamental building block of a vector mapping machine
    (aka "artificial neural network").
    """
    weights: List[float]
    def __init__(self, inputSize:int):
        self.inputSize = inputSize
        self.weights = [0] * (inputSize + 1)
        self.outputSize = 1
    def __weightedSum(self, inputVector:List[float]):
        """This is the "activation function" in AI/ML parlance."""
        result = self.weights[-1] # bias in "AI/ML" parlance
        for i in range(len(self.weights)-1):
            result += self.weights[i] * inputVector[i]
        return result
    def __scaleOutput(self, weightedSum:float):
        """This is the "transfer function" in AI/ML parlance."""
        return 1.0 / (1.0 + exp(-weightedSum))
    def map(self, inputs:List[float]) -> List[float]:
        """Computes the "output of a neuron" in "AI/ML" parlance """
        if self.inputSize != len(inputs):
            raise Exception("Invalid input vector")
        weightedSum = self.__weightedSum(inputs)
        return [self.__scaleOutput(weightedSum)]
    @staticmethod
    def scaleError(output:float, error:float) -> float:
        """Applies the transfer derivative to the error on backpropagation."""
        return error * (output * (1.0 - output))

@dataclass
class ParallelVectorMapper(VectorMapper):
    """
    A "layer" in AI/ML parlance.

    It's constituted by an ordered set of N vector reducers that can
    be executed in parallel on the same input vector to produce
    a new vector from a N-dimensional space. 
    """
    reducers: List[VectorReducer]
    def __init__(self, inputSize:int, outputSize:int) -> None:
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.reducers = [VectorReducer(inputSize) for i in range(outputSize)]
    def map(self, inputs:List[float]) -> List[float]:
        if self.inputSize != len(inputs):
            raise Exception("Invalid input vector")
        # keep it simple stupid
        output:List[float] = []
        for reducer in self.reducers:
            output += reducer.map(inputs)
        return output

@dataclass
class VectorMappingMachine(VectorMapper):
    """
    A "neural network" in AI/ML parlance.
    
    A pipeline of ParallelVectorMapper.
    """
    filters:List[ParallelVectorMapper]
    def __init__(self, inputSize:int, filtersSizes:List[int]) -> None:
        self.inputSize = inputSize
        self.outputSize = filtersSizes[-1]
        self.filters = []
        for i in range(len(filtersSizes)):
            self.filters.append(ParallelVectorMapper(inputSize, filtersSizes[i]))
            inputSize = filtersSizes[i]
    def map(self, inputs:List[float]) -> List[float]:
        if self.inputSize != len(inputs):
            raise Exception("Invalid input vector")
        for mapper in self.filters:
            inputs = mapper.map(inputs)
        return inputs

