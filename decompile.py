#!/usr/bin/env python3
"""A decompiler for programs targetting a Vector Mapping Machine.

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
from ef import *
from sources import *
import sys
import bz2
import pickle
from csv import writer


def help() -> None:
    hs = """
decompile.py program.bin decompiled.sources.csv

Decompile program.bin into decompiled.sources.csv
"""
    print(hs)
    sys.exit()

def decompile(binary:VectorMappingMachineExecutable, writer) -> List[Sample]:
    for sampleIndex in reversed(range(len(binary.logs))):
        log = binary.logs[sampleIndex]
        computedOutputs = log.outputs
        computedError = log.errors

        # easy part: reconstruct sample outputs
        sampleOutputs = [computedOutputs[i] + computedError[i] for i in range(len(computedOutputs))]

        # now, going backward for each filter 
        errors = computedError
        outputs = computedOutputs
        for filterIndex in reversed(range(len(binary.filters))):
            inputs = []
            currentFilter = binary.filters[filterIndex]

            # compute inputs given 
            # - the weight variations of the first node, 
            # - errors, 
            # - output derivative and 
            # - samples' weight ("learning rate" in "AI/ML" parlance)
            firstNodeWeights = currentFilter.reducers[0]
            firstNodeOutput = outputs[0]
            scaledError = VectorReducer.scaleError(firstNodeOutput, errors[0])
            weightVariations = log.sampleNodes[filterIndex]
            for weightIndex in range(len(weightVariations)-1): # skip bias
                deltaW:float = weightVariations[weightIndex]
                if deltaW == 0:
                    inputs.append(0)
                else:
                    d = scaledError*binary.samplesWeight
                    inputs.append(deltaW/d)
                    firstNodeWeights.weights[weightIndex] -= deltaW
            firstNodeWeights.weights[weightIndex+1] -= weightVariations[weightIndex+1] # update "bias" weight

            # given computed inputs and all outputs
            #  -> compute weight variations for all other reducers of the filter
            #  -> apply weight variations to all reducers of the filter
            for nodeIndex in range(1, len(currentFilter.reducers)):
                currentNode = currentFilter.reducers[nodeIndex]
                nodeOutput = outputs[nodeIndex]
                scaledError = VectorReducer.scaleError(nodeOutput, errors[nodeIndex])
                for weightIndex in range(len(currentNode.weights)):
                    if weightIndex != len(currentNode.weights) - 1:
                        deltaW = scaledError*inputs[weightIndex]*binary.samplesWeight
                    else: # "bias" in AI/ML parlance
                        deltaW = scaledError*binary.samplesWeight
                    currentNode.weights[weightIndex] -= deltaW
            
            # so we have the filter in the state it was on backprop of errors
            #  -> compute the errors for the previous filter
            previousFilterErrors = [0] * len(inputs)
            for inputIndex in range(len(inputs)):
                for nodeIndex in range(len(currentFilter.reducers)):
                    weights = currentFilter.reducers[nodeIndex].weights
                    nodeOutput = outputs[nodeIndex]
                    scaledError = VectorReducer.scaleError(nodeOutput, errors[nodeIndex])
                    previousFilterErrors[inputIndex] += weights[inputIndex] * scaledError

            # the inputs of this filter are the outputs of the previous filter
            outputs = inputs
            errors = previousFilterErrors

        csvRow = rescale(binary, Sample(inputs, sampleOutputs))
        writer.writerow(csvRow)

def rescale(binary:VectorMappingMachineExecutable, sample:Sample):
    row = []
    for i in range(len(sample.inputs)):
        stat = binary.encodings.inputs[i]
        if stat.minimum == stat.maximum:
            row.append(round(stat.minimum))
        else:
            rescaled = stat.minimum + sample.inputs[i] * (stat.maximum - stat.minimum)
            if rescaled < 0:
                rescaled *= -1
            row.append(round(rescaled))
    row.append(binary.encodings.outputs[sample.outputs.index(max(sample.outputs))])
    return row

def main(argv:list):
    if len(argv) != 3:
        help()
    with bz2.BZ2File(argv[1], "rb") as f:
        program = pickle.load(f)
    with open(argv[2], "w", newline='') as f:
        csv = writer(f)
        decompile(program, csv)


if __name__ == '__main__':
    main(sys.argv)
