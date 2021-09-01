#!/usr/bin/env python3
"""A compiler targetting Vector Mapping Machines. Compiles source samples in CSV format.

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

from vmm import *
from sources import *
from ef import *

from typing import List, Tuple
import sys
import bz2
import pickle
from random import random, shuffle
from dataclasses import dataclass
from csv import reader

@dataclass
class VectorReducerWrapper(VectorMapper):
    target:VectorReducer
    samplesWeight: float
    weightVariations: List[float]
    output: float
    correction: float
    
    def __init__(self, target:VectorReducer, samplesWeight: float) -> None:
        self.target = target
        self.samplesWeight = samplesWeight # learning rate in "AI/ML" parlance
        self.weightVariations = [float(0)] * len(target.weights)
        self.output = 0.0
        self.correction = 0.0
    def map(self, inputVector:list):
        out = self.target.map(inputVector)
        self.output = out[0]
        return out
    # compute the delta to apply to weights to improve the approximation
    def computeCorrection(self, error:float):
        self.correction = self.target.scaleError(self.output, error)
    # actually update the weights to better approximate the output given the input
    def updateWeights(self, inputVector:list):
        if len(self.target.weights) - 1 != len(inputVector):
            raise Exception("Invalid input vector")
        for i in range(len(inputVector)):
            self.weightVariations[i] = self.samplesWeight * self.correction * inputVector[i] 
            self.target.weights[i] += self.weightVariations[i]
        self.weightVariations[-1] = self.samplesWeight * self.correction
        self.target.weights[-1] += self.weightVariations[-1]


@dataclass
class ParallelVectorMapperWrapper(VectorMapper):
    target:ParallelVectorMapper
    def __init__(self, target:ParallelVectorMapper, samplesWeight: float) -> None:
        self.inputSize = target.inputSize
        self.outputSize = target.outputSize
        self.mappers = [VectorReducerWrapper(v, samplesWeight) for v in target.reducers]
        self.target = target
    def map(self, inputs:List[float]) -> List[float]:
        if self.inputSize != len(inputs):
            raise Exception("Invalid input vector")
        output:List[float] = []
        for mapper in self.mappers:
            output += mapper.map(inputs)
        return output
    def computeCorrection(self, errors:List[float]) -> None:
        if len(errors) != self.outputSize:
            raise Exception("Invalid errors vector")
        for i in range(self.outputSize):
            mapper:VectorReducerWrapper = self.mappers[i]
            mapper.computeCorrection(errors[i])
    def computeInputErrors(self) -> List[float]: # for backward propagation
        errors =  [float(0)] * self.inputSize
        for mapper in self.mappers:
            for i in range(self.inputSize):
                errors[i] += mapper.target.weights[i] * mapper.correction
        return errors
    def getLastOutputs(self) -> List[float]:
        return [r.output for r in self.mappers]
    def updateWeights(self, inputVector:list) -> None:
        if self.inputSize != len(inputVector):
            raise Exception("Invalid input vector")
        for i in range(len(self.mappers)):
            mapper:VectorReducerWrapper = self.mappers[i]
            mapper.updateWeights(inputVector)

def help() -> None:
    hs = """
compile.py machine.vm sources.csv program.bin

Compile sources.csv for machine.vm to program.bin

    machine.vm          The target machine that will run the program
    source.csv          Sources: samples that will be used to 
                        statistically program the machine
    program.bin         Output binary program
"""
    print(hs)
    sys.exit()

# assumes the last column contains the output categorization
def loadSamples(csvfile:str, inputSize:int) -> Tuple[List[Sample], SourceStats]:
    data = []
    outputCategories = []
    rowNum = 0
    samples = []
    with open(csvfile, 'r') as file:
        csv = reader(file)
        for row in csv:
            if not row:
                continue
            if len(row) != inputSize + 1:
                print("Discarding input row %s: incompatible with machine input vector" % rowNum)
                print("InputSize = %s; row = %s" % (inputSize, row))
                continue
            if row[-1] not in outputCategories:
                outputCategories.append(row[-1])
            line = [float(x.strip()) for x in row[:-1]]
            line.append(row[-1])
            data.append(line)
            rowNum += 1
    
    stats = SourceStats([
        ColumnStats(min(column), max(column)) for column in zip(*data)
    ], outputCategories)

    # normalize data
    for line in data:
        for i in range(len(line)-1):
            if stats.inputs[i].maximum > stats.inputs[i].minimum:
                line[i] = (line[i] - stats.inputs[i].minimum) / (stats.inputs[i].maximum - stats.inputs[i].minimum)
            else:
                if stats.inputs[i].minimum == 0:
                    line[i] = 0
                else:
                    line[i] = 1

    # build samples (with one-hot categorical output)
    for line in data:
        output = [0] * len(outputCategories)
        for i in range(len(outputCategories)):
            if outputCategories[i] == line[-1]:
                output[i] = 1
                break
        samples.append(Sample(line[:-1], output))

    return (samples, stats)

def runProgram(program:List[ParallelVectorMapperWrapper], inputs:List[float]) -> List[float]:
    for mapper in program:
        inputs = mapper.map(inputs)
    return inputs

def compile(targetMachine:VectorMappingMachine, samples:List[Sample], sampleWeight:float, epochs:int) -> Tuple[List[ParallelVectorMapper], List[CompilationLog]]:
    logs:List[CompilationLog] = []
    program = [ParallelVectorMapperWrapper(t, sampleWeight) for t in targetMachine.filters]
    for epoch in reversed(range(epochs)):
        for sampleIndex in reversed(range(len(samples))):
            sample = samples[sampleIndex]
            inputs = sample.inputs
            computed = runProgram(program, inputs)
            outputErrors = [sample.outputs[i] - computed[i] for i in range(len(computed))]
            errors = outputErrors
            for filter in reversed(program):
                filter.computeCorrection(errors)
                errors = filter.computeInputErrors()
            for filter in program:
                filter.updateWeights(inputs)
                inputs = filter.getLastOutputs()
            if epoch == 0:
                weightVariations = [l.mappers[0].weightVariations[:] for l in program]
                log = CompilationLog(computed, outputErrors, weightVariations)
                logs.append(log)
                if sampleIndex == 0:
                    return ([w.target for w in program], logs)


def main(argv:list):
    if len(argv) != 4:
        help()
    with open(argv[1], "rb") as f:
        vm = pickle.load(f)
    epochs = 10
    samplesWeight = 0.2     # "learning rate" in AI/ML parlance
    (samples, stats) = loadSamples(argv[2], vm.inputSize)
    (filters, logs) = compile(vm, samples, samplesWeight, epochs)
    filtersDumps = []
    for l in filters:
        reducersDump = [VectorReducerDump(r.inputSize, r.outputSize, r.weights[:]) for r in l.reducers]
        dump = FilterDump(l.inputSize, l.outputSize, reducersDump)
        filtersDumps.append(dump)
    binary = VectorMappingMachineExecutable(vm.inputSize, samplesWeight, stats, filtersDumps, logs)
    with bz2.BZ2File(argv[3], "wb") as f:
        pickle.dump(binary, f)
 
    
if __name__ == '__main__':
    main(sys.argv)
