from typing import List, Optional, Union

import numpy as np
from distances._utils import (
    _reshape_pairwise_single,
    _reshape_to_numba_list,
    reshape_pairwise_to_multiple,
    _reshape_to_numba_list_unjit
)
from numba import njit
from numba.typed import List as NumbaList

from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._squared import _univariate_squared_distance
from aeon.utils.conversion import convert_collection


@njit(cache=True, fastmath=True)
def msm_pairwise_distance_main(
    X: np.ndarray,
    y: np.ndarray = None,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: float = None,
) -> np.ndarray:
    if y is None:
        # To self
        if X.ndim == 3:
            return _msm_pairwise_distance(X, window, independent, c, itakura_max_slope)
        if X.ndim == 2:
            _X = X.reshape((X.shape[0], 1, X.shape[1]))
            return _msm_pairwise_distance(_X, window, independent, c, itakura_max_slope)
        raise ValueError("x and y must be 2D or 3D arrays")
    elif y.ndim == X.ndim:
        # Multiple to multiple
        if y.ndim == 3 and X.ndim == 3:
            return _msm_from_multiple_to_multiple_distance(
                X, y, window, independent, c, itakura_max_slope
            )
        if y.ndim == 2 and X.ndim == 2:
            _x = X.reshape((X.shape[0], 1, X.shape[1]))
            _y = y.reshape((y.shape[0], 1, y.shape[1]))
            return _msm_from_multiple_to_multiple_distance(
                _x, _y, window, independent, c, itakura_max_slope
            )
        if y.ndim == 1 and X.ndim == 1:
            _x = X.reshape((1, 1, X.shape[0]))
            _y = y.reshape((1, 1, y.shape[0]))
            return _msm_from_multiple_to_multiple_distance(
                _x, _y, window, independent, c, itakura_max_slope
            )
        raise ValueError("x and y must be 1D, 2D, or 3D arrays")
    _x, _y = reshape_pairwise_to_multiple(X, y)
    return _msm_from_multiple_to_multiple_distance(
        _x, _y, window, independent, c, itakura_max_slope
    )


def msm_pairwise_distance_two_func(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: float = None,
) -> np.ndarray:
    if y is None:
        # To self
        if isinstance(X, List):
            return _msm_pairwise_distance_list(
                NumbaList(X), window, independent, c, itakura_max_slope
            )
        if isinstance(X, np.ndarray):
            if X.ndim == 3:
                return _msm_pairwise_distance(
                    X, window, independent, c, itakura_max_slope
                )
            if X.ndim == 2:
                _X = X.reshape((X.shape[0], 1, X.shape[1]))
                return _msm_pairwise_distance(
                    _X, window, independent, c, itakura_max_slope
                )

        raise ValueError("x and y must be 2D or 3D arrays")

    if isinstance(X, List) or isinstance(y, List):
        return _msm_from_multiple_to_multiple_distance_list(
            NumbaList(X), NumbaList(y), window, independent, c, itakura_max_slope
        )
    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        if y.ndim == X.ndim:
            # Multiple to multiple
            if y.ndim == 3 and X.ndim == 3:
                return _msm_from_multiple_to_multiple_distance(
                    X, y, window, independent, c, itakura_max_slope
                )
            if y.ndim == 2 and X.ndim == 2:
                _x = X.reshape((X.shape[0], 1, X.shape[1]))
                _y = y.reshape((y.shape[0], 1, y.shape[1]))
                return _msm_from_multiple_to_multiple_distance(
                    _x, _y, window, independent, c, itakura_max_slope
                )
            if y.ndim == 1 and X.ndim == 1:
                _x = X.reshape((1, 1, X.shape[0]))
                _y = y.reshape((1, 1, y.shape[0]))
                return _msm_from_multiple_to_multiple_distance(
                    _x, _y, window, independent, c, itakura_max_slope
                )
            raise ValueError("x and y must be 1D, 2D, or 3D arrays")

        _x, _y = reshape_pairwise_to_multiple(X, y)
        return _msm_from_multiple_to_multiple_distance(
            _x, _y, window, independent, c, itakura_max_slope
        )

    raise ValueError(
        "x and y must have a compatible type (either np.ndarray or List[np.ndarray])!"
    )


def msm_pairwise_distance_only_list(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: float = None,
) -> np.ndarray:
    if y is None:
        # To self
        if isinstance(X, List):
            return _msm_pairwise_distance_list(
                NumbaList(X), window, independent, c, itakura_max_slope
            )

        if isinstance(X, np.ndarray):
            if X.ndim == 3:
                _X: List[np.ndarray] = convert_collection(X, "np-list")
                return _msm_pairwise_distance_list(
                    NumbaList(_X), window, independent, c, itakura_max_slope
                )
            if X.ndim == 2:
                _X: np.ndarray = X.reshape((X.shape[0], 1, X.shape[1]))
                _X2: List[np.ndarray] = convert_collection(_X, "np-list")
                return _msm_pairwise_distance_list(
                    NumbaList(_X2), window, independent, c, itakura_max_slope
                )

        raise ValueError("X must be 2D or 3D")

    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        _X, _y = reshape_pairwise_to_multiple(X, y)
        return _msm_from_multiple_to_multiple_distance_list(
            NumbaList(_X), NumbaList(_y), window, independent, c, itakura_max_slope
        )

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

    return _msm_from_multiple_to_multiple_distance_list(
        NumbaList(_X), NumbaList(_y), window, independent, c, itakura_max_slope
    )


def msm_pairwise_distance_custom_only_list(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: float = None,
) -> np.ndarray:
    _X = _reshape_to_numba_list(X, "X")

    if y is None:
        # To self
        return _msm_pairwise_distance_custom_list(
            _X, window, independent, c, itakura_max_slope
        )

    _y = _reshape_to_numba_list(y, "y")
    return _msm_from_multiple_to_multiple_distance_custom_list(
        _X, _y, window, independent, c, itakura_max_slope
    )


def msm_pairwise_distance_custom_unjit(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: float = None,
    independent: bool = True,
    c: float = 1.0,
    itakura_max_slope: float = None,
) -> np.ndarray:
    _X = _reshape_to_numba_list_unjit(X, "X")

    if y is None:
        # To self
        return _msm_pairwise_distance_custom_list(
            _X, window, independent, c, itakura_max_slope
        )

    _y = _reshape_to_numba_list_unjit(y, "y")
    return _msm_from_multiple_to_multiple_distance_custom_list(
        _X, _y, window, independent, c, itakura_max_slope
    )


@njit(cache=True, fastmath=True)
def _msm_pairwise_distance(
    X: np.ndarray,
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = X.shape[0]
    distances = np.zeros((n_instances, n_instances))
    bounding_matrix = create_bounding_matrix(
        X.shape[2], X.shape[2], window, itakura_max_slope
    )

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            distances[i, j] = _msm_distance(X[i], X[j], bounding_matrix, independent, c)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _msm_pairwise_distance_list(
    X: NumbaList[np.ndarray],
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = len(X)
    distances = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            x1, x2 = _reshape_pairwise_single(X[i], X[j])
            bounding_matrix = create_bounding_matrix(
                x1.shape[-1], x2.shape[-1], window, itakura_max_slope
            )
            distances[i, j] = _msm_distance(x1, x2, bounding_matrix, independent, c)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _msm_pairwise_distance_custom_list(
    X: NumbaList[np.ndarray],
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = len(X)
    distances = np.zeros((n_instances, n_instances))

    if window == 1:
        max_shape = max([x.shape[-1] for x in X])
        bounding_matrix: np.ndarray = create_bounding_matrix(
            max_shape, max_shape, window, itakura_max_slope
        )
    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            x1, x2 = X[i], X[j]
            if window != 1:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[-1], x2.shape[-1], window, itakura_max_slope
                )
            distances[i, j] = _msm_distance(x1, x2, bounding_matrix, independent, c)
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True)
def _msm_from_multiple_to_multiple_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = x.shape[0]
    m_instances = y.shape[0]
    distances = np.zeros((n_instances, m_instances))
    bounding_matrix = create_bounding_matrix(
        x.shape[2], y.shape[2], window, itakura_max_slope
    )

    for i in range(n_instances):
        for j in range(m_instances):
            distances[i, j] = _msm_distance(x[i], y[j], bounding_matrix, independent, c)
    return distances


@njit(cache=True, fastmath=True)
def _msm_from_multiple_to_multiple_distance_list(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = len(x)
    m_instances = len(y)
    distances = np.zeros((n_instances, m_instances))

    for i in range(n_instances):
        for j in range(m_instances):
            x1, y1 = _reshape_pairwise_single(x[i], y[j])
            bounding_matrix = create_bounding_matrix(
                x1.shape[-1], y1.shape[-1], window, itakura_max_slope
            )
            distances[i, j] = _msm_distance(x1, y1, bounding_matrix, independent, c)
    return distances


@njit(cache=True, fastmath=True)
def _msm_from_multiple_to_multiple_distance_custom_list(
    X: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float,
    independent: bool,
    c: float,
    itakura_max_slope: float,
) -> np.ndarray:
    n_instances = len(X)
    m_instances = len(y)
    distances = np.zeros((n_instances, m_instances))

    if window == 1:
        max_shape = max([x.shape[-1] for x in X])
        bounding_matrix: np.ndarray = create_bounding_matrix(
            max_shape, max_shape, window, itakura_max_slope
        )
    for i in range(n_instances):
        for j in range(m_instances):
            x1, y1 = X[i], y[j]
            if window != 1:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[-1], y1.shape[-1], window, itakura_max_slope
                )
            distances[i, j] = _msm_distance(x1, y1, bounding_matrix, independent, c)
    return distances


@njit(cache=True, fastmath=True)
def _msm_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    independent: bool,
    c: float,
) -> float:
    if independent:
        return _msm_independent_cost_matrix(x, y, bounding_matrix, c)[
            x.shape[1] - 1, y.shape[1] - 1
        ]
    return _msm_dependent_cost_matrix(x, y, bounding_matrix, c)[
        x.shape[1] - 1, y.shape[1] - 1
    ]


@njit(cache=True, fastmath=True)
def _msm_independent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    distance = 0
    for i in range(x.shape[0]):
        curr_cost_matrix = _independent_cost_matrix(x[i], y[i], bounding_matrix, c)
        cost_matrix = np.add(cost_matrix, curr_cost_matrix)
        distance += curr_cost_matrix[-1, -1]
    return cost_matrix


@njit(cache=True, fastmath=True)
def _independent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float
) -> np.ndarray:
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 0] = np.abs(x[0] - y[0])

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost = _cost_independent(x[i], x[i - 1], y[0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost

    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            cost = _cost_independent(y[i], y[i - 1], x[0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                d1 = cost_matrix[i - 1][j - 1] + np.abs(x[i] - y[j])
                d2 = cost_matrix[i - 1][j] + _cost_independent(x[i], x[i - 1], y[j], c)
                d3 = cost_matrix[i][j - 1] + _cost_independent(y[j], x[i], y[j - 1], c)

                cost_matrix[i, j] = min(d1, d2, d3)

    return cost_matrix


@njit(cache=True, fastmath=True)
def _cost_independent(x: float, y: float, z: float, c: float) -> float:
    if (y <= x <= z) or (y >= x >= z):
        return c
    return c + min(abs(x - y), abs(x - z))


@njit(cache=True, fastmath=True)
def _msm_dependent_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, c: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 0] = np.sum(np.abs(x[:, 0] - y[:, 0]))

    for i in range(1, x_size):
        if bounding_matrix[i, 0]:
            cost = _cost_dependent(x[:, i], x[:, i - 1], y[:, 0], c)
            cost_matrix[i][0] = cost_matrix[i - 1][0] + cost
    for i in range(1, y_size):
        if bounding_matrix[0, i]:
            cost = _cost_dependent(y[:, i], y[:, i - 1], x[:, 0], c)
            cost_matrix[0][i] = cost_matrix[0][i - 1] + cost

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i, j]:
                d1 = cost_matrix[i - 1][j - 1] + np.sum(np.abs(x[:, i] - y[:, j]))
                d2 = cost_matrix[i - 1][j] + _cost_dependent(
                    x[:, i], x[:, i - 1], y[:, j], c
                )
                d3 = cost_matrix[i][j - 1] + _cost_dependent(
                    y[:, j], x[:, i], y[:, j - 1], c
                )

                cost_matrix[i, j] = min(d1, d2, d3)
    return cost_matrix


@njit(cache=True, fastmath=True)
def _cost_dependent(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: float) -> float:
    diameter = _univariate_squared_distance(y, z)
    mid = (y + z) / 2
    distance_to_mid = _univariate_squared_distance(mid, x)

    if distance_to_mid <= (diameter / 2):
        return c
    else:
        dist_to_q_prev = _univariate_squared_distance(y, x)
        dist_to_c = _univariate_squared_distance(z, x)
        if dist_to_q_prev < dist_to_c:
            return c + dist_to_q_prev
        else:
            return c + dist_to_c
