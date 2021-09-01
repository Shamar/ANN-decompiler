"""The equivalent of an Abstract Syntax Tree for a compiler targetting a Vector Mapping Machine.

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
class Sample:
    """A normalized source sample to program a Vector Mapper."""
    inputs: List[float]
    outputs: List[int]
