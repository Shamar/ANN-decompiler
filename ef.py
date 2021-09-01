"""The executable format for a Vector Mapping Machine.

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

from dataclasses import dataclass
from typing import List

@dataclass
class ColumnStats:
    minimum: float
    maximum: float

@dataclass
class SourceStats:
    inputs:List[ColumnStats]
    outputs:List[str]           # names of categories

@dataclass
class VariationOfWeights:
    variations:List[float]

@dataclass
class CompilationLog:
    outputs:List[float]
    errors:List[float]
    sampleNodes:List[VariationOfWeights] # from one reducer for each filter of the program 

@dataclass
class VectorReducerDump:
    inputSize:int
    outputSize:int
    weights: List[float]

@dataclass
class FilterDump:
    inputSize:int
    outputSize:int
    reducers: List[VectorReducerDump]

@dataclass
class VectorMappingMachineExecutable:
    """A program for a VectorReducingMachine"""
    inputSize:int
    samplesWeight:float         # "learning rate" in AI/ML parlance
    encodings:SourceStats
    filters:List[FilterDump]    # final state of the virtual machine
    logs:List[CompilationLog]   # one for each sample in the last epoch
