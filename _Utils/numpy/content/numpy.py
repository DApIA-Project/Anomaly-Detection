"""Numpy overload with type hinting"""

from . import typing as np
from typing import overload, TypeVar


T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')

@overload
def concatenate(arrays: "list[np.array_1d[T, T1]]") -> np.array_1d[T, T1]: ...


from numpy import *

max = amax
min = amin



