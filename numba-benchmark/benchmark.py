import time
from timeit import Timer
from typing import List

import numpy as np
import pandas as pd
from distances import (
    msm_pairwise_distance_custom_only_list,
    msm_pairwise_distance_main,
    msm_pairwise_distance_only_list,
    msm_pairwise_distance_two_func,
    sbd_pairwise_distance_custom_only_list,
    sbd_pairwise_distance_main,
    sbd_pairwise_distance_only_list,
    sbd_pairwise_distance_two_funcs,
)


def _supports_nonequal_length(dist_func) -> bool:
    anns = dist_func.__annotations__
    return any(param in anns and str(List) in str(anns[param]) for param in ["x", "X"])


def _timeit(func, repeat: int = 3) -> float:
    timer = Timer(func)
    number, time_taken = timer.autorange()
    raw_timings = timer.repeat(repeat=repeat - 1, number=number)
    raw_timings += [time_taken]
    timings = [dt / number for dt in raw_timings]
    best = min(timings)
    print(f"{number} loops, best of {repeat}: {best:.6f}s per loop")
    return best


def main():
    # TODO: becnhmark
    # - run it 2-3 times first
    # - then, benchmark with 1000, 5000, up to 25000 of length 100
    rng = np.random.default_rng(42)
    distance_funcs = [
        sbd_pairwise_distance_main,
        sbd_pairwise_distance_two_funcs,
        sbd_pairwise_distance_only_list,
        sbd_pairwise_distance_custom_only_list,
        msm_pairwise_distance_main,
        msm_pairwise_distance_two_func,
        msm_pairwise_distance_only_list,
        msm_pairwise_distance_custom_only_list,
    ]
    # max_timepoints_order = 5
    # timepoints_options = [int(10**i) for i in range(1, max_timepoints_order)]
    timepoints_options = [10, 100, 1000, 10000, 25000]
    # max_instance_order = 3
    # instance_options = [int(10**i) for i in range(1, max_instance_order)]
    instance_options = [1, 10, 100, 1000, 5000, 25000]

    # warmup for Numba JIT
    print("Warming up Numba JIT...")
    ts1 = rng.random(100)
    ts2 = rng.random(100)
    for i in [1, 2, 5, 10]:
        for func in distance_funcs:
            func(ts1.reshape(i, 1, -1), ts2.reshape(i, 1, -1))
            func(ts1.reshape(i, 2, -1), ts2.reshape(i, 2, -1))

    ts1 = [rng.random(100) for _ in range(5)]
    ts2 = [rng.random(100) for _ in range(5)]
    ts1_multi = [rng.random((2, 50)) for _ in range(5)]
    ts2_multi = [rng.random((2, 50)) for _ in range(5)]
    for func in distance_funcs:
        if _supports_nonequal_length(func):
            func(ts1, ts2)
            func(ts1_multi, ts2_multi)

    time.sleep(2)
    print("...done.")

    print("Starting benchmark (univariate)...")
    results = []
    for func in distance_funcs:
        print(f"  {func.__name__}:")
        for n_timepoints in timepoints_options:
            n_instances = 1
            n_channels = 2
            ts1 = rng.random((n_channels, n_timepoints))
            ts2 = rng.random((n_channels, n_timepoints))
            print(f"    input=({n_instances}, {n_channels}, {n_timepoints}): ", end="")
            best = _timeit(lambda: func(ts1, ts2))
            results.append(
                {
                    "distance": func.__name__,
                    "n_instances": n_instances,
                    "n_channels": n_channels,
                    "n_timepoints_min": n_timepoints,
                    "n_timepoints_max": n_timepoints,
                    "time": best,
                }
            )
        pd.DataFrame(results).to_csv("numba-benchmark.bak.csv", index=False)

        for n_instances in instance_options:
            n_channels = 1
            n_timepoints = 10
            ts1 = rng.random((n_instances, n_channels, n_timepoints))
            ts2 = rng.random((n_instances, n_channels, n_timepoints))
            print(f"    input=({n_instances}, {n_channels}, {n_timepoints}): ", end="")
            best = _timeit(lambda: func(ts1, ts2))
            results.append(
                {
                    "distance": func.__name__,
                    "n_instances": n_instances,
                    "n_channels": n_channels,
                    "n_timepoints_min": n_timepoints,
                    "n_timepoints_max": n_timepoints,
                    "time": best,
                }
            )

        if _supports_nonequal_length(func):
            for n_instances in instance_options:
                n_channels = 1
                n_timepoints = 10
                ts1 = [
                    rng.random(n_timepoints + i % 11 - 5) for i in range(n_instances)
                ]
                ts2 = [
                    rng.random(n_timepoints + i % 11 - 5) for i in range(n_instances)
                ]
                print(
                    f"    input=({n_instances}, {{{n_timepoints-5}, {n_timepoints+5}}}): ",
                    end="",
                )
                best = _timeit(lambda: func(ts1, ts2))
                results.append(
                    {
                        "distance": func.__name__,
                        "n_instances": n_instances,
                        "n_channels": n_channels,
                        "n_timepoints_min": n_timepoints - 5,
                        "n_timepoints_max": n_timepoints + 5,
                        "time": best,
                    }
                )
        pd.DataFrame(results).to_csv("numba-benchmark.bak.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("numba-benchmark.csv", index=False)
    print("...done.")


if __name__ == "__main__":
    main()
