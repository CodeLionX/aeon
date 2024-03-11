"""Shape-based distance (SBD) between two time series."""

__maintainer__ = ["codelionx"]

from typing import List, Optional, Union

import numpy as np
from distances._utils import (
    _reshape_pairwise_single,
    _reshape_to_numba_list,
    reshape_pairwise_to_multiple,
)
from numba import njit, objmode
from numba.typed import List as NumbaList
from scipy.signal import correlate

from aeon.utils.conversion import convert_collection


@njit(cache=True, fastmath=True)
def sbd_pairwise_distance_main(
    x: np.ndarray, y: Optional[np.ndarray] = None, standardize: bool = True
) -> np.ndarray:
    if y is None:
        # To self
        if x.ndim == 3:
            return _sbd_pairwise_distance_single(x, standardize)
        elif x.ndim == 2:
            _X = x.reshape((x.shape[0], 1, x.shape[1]))
            return _sbd_pairwise_distance_single(_X, standardize)
        raise ValueError("X must be 2D or 3D")

    _x, _y = reshape_pairwise_to_multiple(x, y)
    return _sbd_pairwise_distance(_x, _y, standardize)


def sbd_pairwise_distance_two_funcs(
    x: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    standardize: bool = True,
) -> np.ndarray:
    if y is None:
        # To self
        if isinstance(x, List):
            _x = NumbaList(x)
            # _ensure_equal_dims_in_list(_x)
            return _sbd_pairwise_distance_single_list(_x, standardize)

        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                return _sbd_pairwise_distance_single(x, standardize)
            if x.ndim == 2:
                _X = x.reshape((x.shape[0], 1, x.shape[1]))
                return _sbd_pairwise_distance_single(_X, standardize)

        raise ValueError("X must be 2D or 3D")

    if isinstance(x, List) and isinstance(y, List):
        _x = NumbaList(x)
        _y = NumbaList(y)
        # _ensure_equal_dims_in_list(_x, _y)
        return _sbd_pairwise_distance_list(_x, _y, standardize)

    if isinstance(x, List) and isinstance(y, np.ndarray):
        _x = NumbaList(x)
        _y = NumbaList(y)
        # _ensure_equal_dims_in_list(_x)
        return _sbd_pairwise_distance_list(_x, _y, standardize)

    if isinstance(x, np.ndarray) and isinstance(y, List):
        _x = NumbaList(x)
        _y = NumbaList(y)
        # _ensure_equal_dims_in_list(_y)
        return _sbd_pairwise_distance_list(_x, _y, standardize)

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        _x, _y = reshape_pairwise_to_multiple(x, y)
        # _ensure_equal_dims(_x, _y)
        return _sbd_pairwise_distance(_x, _y, standardize)

    raise ValueError(
        "x and y must have a compatible type (either np.ndarray or List[np.ndarray])!"
    )


def sbd_pairwise_distance_only_list(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    standardize: bool = True,
) -> np.ndarray:
    if y is None:
        # To self
        if isinstance(X, List):
            return _sbd_pairwise_distance_single_list(NumbaList(X), standardize)

        if isinstance(X, np.ndarray):
            if X.ndim == 3:
                _X: List[np.ndarray] = convert_collection(X, "np-list")
                return _sbd_pairwise_distance_single_list(NumbaList(_X), standardize)
            if X.ndim == 2:
                _X: np.ndarray = X.reshape((X.shape[0], 1, X.shape[1]))
                _X2: List[np.ndarray] = convert_collection(_X, "np-list")
                return _sbd_pairwise_distance_single_list(NumbaList(_X2), standardize)

        raise ValueError("X must be 2D or 3D")

    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        _X, _y = reshape_pairwise_to_multiple(X, y)
        return _sbd_pairwise_distance_list(NumbaList(_X), NumbaList(_y), standardize)

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            _X: List[np.ndarray] = [X.reshape(1, -1)]
        else:
            _X = convert_collection(X, "np-list")
    elif isinstance(X, List):
        _X = X
    else:
        raise ValueError("x must be either np.ndarray or List[np.ndarray]")

    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            _y: List[np.ndarray] = [y.reshape(1, -1)]
        else:
            _y = convert_collection(y, "np-list")
    elif isinstance(y, List):
        _y = y
    else:
        raise ValueError("y must be either np.ndarray or List[np.ndarray]")

    return _sbd_pairwise_distance_list(NumbaList(_X), NumbaList(_y), standardize)


def sbd_pairwise_distance_custom_only_list(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    standardize: bool = True,
) -> np.ndarray:
    _X = _reshape_to_numba_list(X, "X")

    if y is None:
        # To self
        return _sbd_pairwise_distance_single_custom_list(_X, standardize)

    _y = _reshape_to_numba_list(y, "y")
    return _sbd_pairwise_distance_custom_list(_X, _y, standardize)


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_single(x: np.ndarray, standardize: bool) -> np.ndarray:
    n_instances = x.shape[0]
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = sbd_distance(x[i], x[j], standardize)
            distances[j, i] = distances[i, j]
    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_single_list(
    x: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_instances = len(x)
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            ts1, ts2 = _reshape_pairwise_single(x[i], x[j])
            distances[i, j] = sbd_distance(ts1, ts2, standardize)
            distances[j, i] = distances[i, j]
    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_single_custom_list(
    x: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_instances = len(x)
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = sbd_distance(x[i], x[j], standardize)
            distances[j, i] = distances[i, j]
    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance(
    x: np.ndarray, y: np.ndarray, standardize: bool
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = sbd_distance(x[i], y[j], standardize)
    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_list(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_instances = len(x)
    m_instances = len(y)
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            ts1, ts2 = _reshape_pairwise_single(x[i], y[j])
            distances[i, j] = sbd_distance(ts1, ts2, standardize)
    return distances


@njit(cache=True, fastmath=True)
def _sbd_pairwise_distance_custom_list(
    x: NumbaList[np.ndarray], y: NumbaList[np.ndarray], standardize: bool
) -> np.ndarray:
    n_instances = len(x)
    m_instances = len(y)
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = sbd_distance(x[i], y[j], standardize)
    return distances


@njit(cache=True, fastmath=True)
def sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_sbd_distance(x, y, standardize)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == y.shape[0] == 1:
            _x = x.ravel()
            _y = y.ravel()
            return _univariate_sbd_distance(_x, _y, standardize)
        else:
            # independent
            nchannels = min(x.shape[0], y.shape[0])
            distance = 0.0
            for i in range(nchannels):
                distance += _univariate_sbd_distance(x[i], y[i], standardize)
            return distance / nchannels

    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _univariate_sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if standardize:
        if x.size == 1 or y.size == 1:
            return 0.0

    try:
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    except Exception:
        print("Zero division error", x, y)
        raise ZeroDivisionError()

    with objmode(a="float64[:]"):
        a = correlate(x, y, method="fft")

    b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.abs(1.0 - np.max(a / b))
