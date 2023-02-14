import pathlib
import time
import pickle
import math
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal

@dataclass
class Hello:
    x: Any
    y: Any

yay = Hello(1,2)
print(type(yay))
print(type(yay) == Hello)