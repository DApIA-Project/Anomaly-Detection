from typing import Generic, TypeVar
import numpy as np

np.ndarray

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
S = TypeVar('S')

class NP:
    class float32_4d(Generic[T1, T2, T3, T4]):
        def __getitem__(self, key:object) -> np.float32:
            return super().__getitem__(key)
    class float64_4d(Generic[T1, T2, T3, T4]):
        def __getitem__(self, key:object) -> np.float64:
            return super().__getitem__(key)
    class int32_4d(Generic[T1, T2, T3, T4]):
        def __getitem__(self, key:object) -> np.int32:
            return super().__getitem__(key)
    class int64_4d(Generic[T1, T2, T3, T4]):
        def __getitem__(self, key:object) -> np.int64:
            return super().__getitem__(key)
    class nd_4d(Generic[T, T1, T2, T3, T4]):
        def __getitem__(self, key:object) -> T:
            return super().__getitem__(key)

    class float32_3d(Generic[T1, T2, T3]):
        def __getitem__(self, key:object) -> np.float32:
            return super().__getitem__(key)
    class float64_3d(Generic[T1, T2, T3]):
        def __getitem__(self, key:object) -> np.float64:
            return super().__getitem__(key)
    class int32_3d(Generic[T1, T2, T3]):
        def __getitem__(self, key:object) -> np.int32:
            return super().__getitem__(key)
    class int64_3d(Generic[T1, T2, T3]):
        def __getitem__(self, key:object) -> np.int64:
            return super().__getitem__(key)
    class nd_3d(Generic[T, T1, T2, T3]):
        def __getitem__(self, key:object) -> T:
            return super().__getitem__(key)

    class float32_2d(Generic[T1, T2]):
        def __getitem__(self, key:object) -> np.float32:
            return super().__getitem__(key)
    class float64_2d(Generic[T1, T2]):
        def __getitem__(self, key:object) -> np.float64:
            return super().__getitem__(key)
    class int32_2d(Generic[T1, T2]):
        def __getitem__(self, key:object) -> np.int32:
            return super().__getitem__(key)
    class int64_2d(Generic[T1, T2]):
        def __getitem__(self, key:object) -> np.int64:
            return super().__getitem__(key)
    class nd_2d(Generic[T, T1, T2]):
        def __getitem__(self, key:object) -> T:
            return super().__getitem__(key)

    class float32_1d(Generic[T1]):
        def __getitem__(self, key:object) -> np.float32:
            return super().__getitem__(key)
    class float64_1d(Generic[T1]):
        def __getitem__(self, key:object) -> np.float64:
            return super().__getitem__(key)
    class int32_1d(Generic[T1]):
        def __getitem__(self, key:object) -> np.int32:
            return super().__getitem__(key)
    class int64_1d(Generic[T1]):
        def __getitem__(self, key:object) -> np.int64:
            return super().__getitem__(key)
    class nd_1d(Generic[T, T1]):
        def __getitem__(self, key:object) -> T:
            return super().__getitem__(key)



class AX:
    # machine learning
    class batch: pass
    class sample: pass
    class time: pass
    class feature: pass

    # coordinates
    class x: pass
    class y: pass
    class z: pass

    # colors
    class rgb: pass
    class rgba: pass


