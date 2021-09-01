#!/usr/bin/env python3
"""A tool to test the accuracy of a program targetting Vector Mapping Machines.

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
from typing import List
import sys
import bz2
import pickle
from csv import reader

def loadSamples(csvfile:str, inputSize:int, encodings:SourceStats) -> List[Sample]:
    data = []
    rowNum = 0
    samples = []
    with open(csvfile, 'r') as file:
        csv = reader(file)
        for row in csv:
            if not row:
                continue
            if len(row) != inputSize + 1:
                print("Discarding test row %s: incompatible with machine input vector" % rowNum)
                continue
            if row[-1] not in encodings.outputs:
                print("Discarding test row %s: unknown category %s" % (rowNum, row[-1]))
                continue
            line = [float(x.strip()) for x in row[:-1]]
            line.append(row[-1])
            data.append(line)
            rowNum += 1
    
    # normalize data
    for line in data:
        for i in range(len(line)-1):
            if encodings.inputs[i].maximum > encodings.inputs[i].minimum:
                line[i] = (line[i] - encodings.inputs[i].minimum) / (encodings.inputs[i].maximum - encodings.inputs[i].minimum)
            else:
                if encodings.inputs[i].minimum == 0:
                    line[i] = 0
                else:
                    line[i] = 1

    # build samples (with one-hot categorical output)
    for line in data:
        output = [0] * len(encodings.outputs)
        for i in range(len(encodings.outputs)):
            if encodings.outputs[i] == line[-1]:
                output[i] = 1
                break
        samples.append(Sample(line[:-1], output))

    return samples


def help() -> None:
    hs = """
test.py machine.vm program.bin testSet.csv

Run program.bin on the provided vector reducing machine, with the 
provided testSet as input and check the output correctness.
"""
    print(hs)
    sys.exit()

def fail(message:str):
    print("ERROR: incompatible program: " + message)
    sys.exit()

def loadProgram(program:VectorMappingMachineExecutable, machine:VectorMappingMachine) -> None:
    if len(program.filters) != len(machine.filters):
        fail("the machine has %s filters, while the program requires %s filters" % (len(machine.filters), len(program.filters)))
    if program.filters[0].inputSize != machine.inputSize:
        fail("the machine transforms vectors of size %s, while the program is for machines transforming vectors of size %s" % (machine.inputSize, machine.filters[0].inputSize))
    if program.filters[-1].outputSize != machine.outputSize:
        fail("the machine produces %s, while the program is for machines producing vectors of size %s" % (machine.outputSize, machine.filters[0].outputSize))
    for l in range(len(program.filters)):
        if machine.filters[l].inputSize != program.filters[l].inputSize:
            fail("incompatible input vector size at filter %s" % l)
        if machine.filters[l].outputSize != program.filters[l].outputSize:
            fail("incompatible output vector size at filter %s" % l)
        if program.filters[l].outputSize != len(program.filters[l].reducers):
            fail("program error: wrong number of reducers at filter %s" % l)
        for r in range(len(program.filters[l].reducers)):
            for w in range(len(program.filters[l].reducers[r].weights)):
                machine.filters[l].reducers[r].weights[w] = program.filters[l].reducers[r].weights[w]

def main(argv:list):
    if len(argv) != 4:
        help()
    with open(argv[1], "rb") as f:
        vm:VectorMappingMachine = pickle.load(f)
    with bz2.BZ2File(argv[2], "rb") as f:
        program = pickle.load(f)
    loadProgram(program, vm)
    samples = loadSamples(argv[3], vm.inputSize, program.encodings)
    
    errors = []
    for sampleIdx in range(len(samples)):
        sample = samples[sampleIdx]
        result = vm.map(sample.inputs)
        if result.index(max(result)) != sample.outputs.index(1):
            errors.append(sample)
    
    print("Correct \"predictions\": %s/%s (%s%%)" %(len(samples) - len(errors), len(samples), 100 * (len(samples) - len(errors)) / float(len(samples)))) 
    
if __name__ == '__main__':
    main(sys.argv)
