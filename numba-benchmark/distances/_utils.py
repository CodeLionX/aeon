from typing import Tuple, List, Union

import numpy as np
from numba import njit
from numba.typed import List as NumbaList


@njit(cache=True, fastmath=True)
def reshape_pairwise_to_multiple(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim == y.ndim:
        if y.ndim == 3 and x.ndim == 3:
            return x, y
        if y.ndim == 2 and x.ndim == 2:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        if y.ndim == 1 and x.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    else:
        if x.ndim == 3 and y.ndim == 2:
            _y = y.reshape((1, y.shape[0], y.shape[1]))
            return x, _y
        if y.ndim == 3 and x.ndim == 2:
            _x = x.reshape((1, x.shape[0], x.shape[1]))
            return _x, y
        if x.ndim == 2 and y.ndim == 1:
            _x = x.reshape((x.shape[0], 1, x.shape[1]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, 1, x.shape[0]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _x, _y
        raise ValueError("x and y must be 2D or 3D arrays")


@njit(cache=True, fastmath=True)
def _reshape_pairwise_single(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim == y.ndim:
        if y.ndim == 2 and x.ndim == 2:
            return x, y
        if y.ndim == 1 and x.ndim == 1:
            _x = x.reshape((1, x.shape[0]))
            _y = y.reshape((1, y.shape[0]))
            return _x, _y
        raise ValueError("x and y must be 1D or 2D arrays")
    else:
        if x.ndim == 2 and y.ndim == 1:
            _y = y.reshape((1, y.shape[0]))
            return x, _y
        if y.ndim == 2 and x.ndim == 1:
            _x = x.reshape((1, x.shape[0]))
            return _x, y
        raise ValueError("x and y must be 1D or 2D arrays")


@njit(cache=True, fastmath=True)
def _reshape_single(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    elif x.ndim == 1:
        return x.reshape((1, x.shape[0]))
    else:
        raise ValueError("x and y must be 1D or 2D arrays")


@njit(cache=True, fastmath=True)
def _reshape_all_in_list(x: NumbaList[np.ndarray]) -> NumbaList[np.ndarray]:
    x_new = NumbaList()
    for i in range(len(x)):
        x_new.append(_reshape_single(x[i]))
    return x_new


@njit(cache=True, fastmath=True)
def _reshape_ndarray_to_list(x: np.ndarray) -> NumbaList[np.ndarray]:
    if x.ndim == 3:
        return NumbaList(x)
    elif x.ndim == 2:
        return NumbaList(x.reshape(x.shape[0], 1, x.shape[1]))
    elif x.ndim == 1:
        return NumbaList(x.reshape(1, 1, -1))
    else:
        raise ValueError("x must be 1D, 2D or 3D")


def _reshape_to_numba_list(X: Union[np.ndarray, List[np.ndarray]], name: str = "X") -> NumbaList[np.ndarray]:
    if isinstance(X, np.ndarray):
        return _reshape_ndarray_to_list(X)
    elif isinstance(X, List):
        return _reshape_all_in_list(NumbaList(X))
    else:
        raise ValueError(f"{name} must be either np.ndarray or List[np.ndarray]")
