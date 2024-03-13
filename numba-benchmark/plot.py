import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    results_path = Path("numba-benchmark.csv")
    if not results_path.exists():
        raise FileNotFoundError("No results found, please either try with a backup or re-run the benchmark!")

    df = pd.read_csv(results_path)
    df["distance_func"] = df["distance"]
    df["distance"] = df["distance"].str.split("_").str[0]
    distances = df["distance"].unique().tolist()
    print(df)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    fig.suptitle("Runtime vs. Time Series Length")
    for i, distance in enumerate(distances):
        df_tmp = df[df["distance"] == distance]
        axs[i].set_title(distance)
        axs[i].set_xlabel("time series length")
        axs[i].set_ylabel("runtime (s)")
        for distance_func in df_tmp["distance_func"].unique():
            mask = (df_tmp["distance_func"] == distance_func) & (df_tmp["n_instances"] == 1) & (df_tmp["n_channels"] == 2)
            axs[i].plot(df_tmp.loc[mask, "n_timepoints_max"], df_tmp.loc[mask, "time"], label=distance_func)
        axs[i].legend()

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    fig.suptitle("Runtime vs. Collection Size (fixed length)")
    for i, distance in enumerate(distances):
        df_tmp = df[df["distance"] == distance]
        axs[i].set_title(distance)
        axs[i].set_xlabel("number of time series")
        axs[i].set_ylabel("runtime (s)")
        for distance_func in df_tmp["distance_func"].unique():
            mask = (df_tmp["distance_func"] == distance_func) & (df_tmp["n_timepoints_max"] == 10) & (df_tmp["n_channels"] == 1)
            axs[i].plot(df_tmp.loc[mask, "n_instances"], df_tmp.loc[mask, "time"], label=distance_func)
        axs[i].legend()

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    fig.suptitle("Runtime vs. Collection Size (variable length)")
    for i, distance in enumerate(distances):
        df_tmp = df[df["distance"] == distance]
        axs[i].set_title(distance)
        axs[i].set_xlabel("number of time series")
        axs[i].set_ylabel("runtime (s)")
        for distance_func in df_tmp["distance_func"].unique():
            mask = (df_tmp["distance_func"] == distance_func) & (df_tmp["n_timepoints_max"] == 15) & (df_tmp["n_channels"] == 1)
            axs[i].plot(df_tmp.loc[mask, "n_instances"], df_tmp.loc[mask, "time"], label=distance_func)
        axs[i].legend()
    plt.show()


if __name__ == '__main__':
    main()
