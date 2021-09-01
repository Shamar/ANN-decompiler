#!/usr/bin/env python3
"""A tool to assemble a Vector Mapping Machine (aka "artificial neural network").

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
from random import random
import sys
import pickle

def help() -> None:
    hs = """
assemble.py machineName.vm InputSize [IntermediateSize ...] OutputSize

Assemble a vector mapping (virtual) machine

    machineName.vm      Output file
    InputSize           Size (integer) of input vector
    IntermediateSize... Sizes (integer) of intermediate vector transformers
    OutputSize          Size (integer) of output vector
"""
    print(hs)
    sys.exit()

def main(argv:list):
    if len(argv) < 4:
        help()
    sizes:List[int] = []
    for i in range(2, len(argv)):
        try:
            sizes.append(int(argv[i]))
        except:
            print("'%s' is not a valid base 10 integer." % argv[i])
            sys.exit()
    vrm = VectorMappingMachine(sizes[0], sizes[1:])
    for l in vrm.filters:
        for r in l.reducers:
            for i in range(len(r.weights)):
                if i % 2 == 0:
                    r.weights[i] = -random()
                else:
                    r.weights[i] = random()
    with open(argv[1], "wb") as f:
        pickle.dump(vrm, f)


if __name__ == '__main__':
    main(sys.argv)
