"""
File: running_stats.py
Author: Matthew Allen

Description:
    An implementation of Welford's algorithm for running statistics.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class WelfordRunningStatStateDict(TypedDict):
    mean: list[float]
    var: list[float]
    shape: tuple[int, ...]
    count: int


class WelfordRunningStat(object):
    """
    https://www.johndcook.com/blog/skewness_kurtosis/
    """

    def __init__(self, shape: tuple[int, ...]):
        self.ones: NDArray[np.float32] = np.ones(shape=shape, dtype=np.float32)
        self.zeros: NDArray[np.float32] = np.zeros(shape=shape, dtype=np.float32)

        self.running_mean: NDArray[np.float32] = np.zeros(shape=shape, dtype=np.float32)
        self.running_variance: NDArray[np.float32] = np.zeros(
            shape=shape, dtype=np.float32
        )

        self.count: int = 0
        self.shape: tuple[int, ...] = shape

    def increment(self, samples: np.ndarray, num: int):
        if num > 1:
            for i in range(num):
                self.update(samples[i])
        else:
            self.update(samples)

    def update(self, sample: np.ndarray):
        current_count = self.count
        self.count += 1
        delta = (sample - self.running_mean).reshape(self.running_mean.shape)
        delta_n = (delta / self.count).reshape(self.running_mean.shape)

        self.running_mean += delta_n
        self.running_variance += delta * delta_n * current_count

    def reset(self):
        del self.running_mean
        del self.running_variance

        self.__init__(self.shape)

    @property
    def mean(self):
        if self.count < 2:
            return self.zeros
        return self.running_mean

    @property
    def std(self) -> NDArray[np.float32]:
        if self.count < 2:
            return self.ones

        var = self.running_variance / (self.count - 1)

        # Wherever variance is zero, set it to 1 to avoid division by zero.
        var = np.where(var == 0, 1.0, var)
        return np.sqrt(var)

    def increment_from_serialized_other(self, serialized_other: np.ndarray):
        """
        Function to combine the statistics computed by two independent instances of this algorithm.
        :param serialized_other: A list containing the serialization of the other instance to be combined with.
        :return: None.
        """
        # Unpack the serialized data.
        n = np.prod(self.shape)
        other_mean = np.asarray(serialized_other[:n], dtype=np.float32).reshape(
            self.running_mean.shape
        )
        other_var = np.asarray(serialized_other[n:-1], dtype=np.float32).reshape(
            self.running_variance.shape
        )
        other_count = serialized_other[-1]
        if other_count == 0:
            return

        # Combine our statistics with the new data.
        count = self.count + other_count

        mean_delta = other_mean - self.running_mean
        mean_delta_squared = mean_delta * mean_delta

        combined_mean = (
            self.count * self.running_mean + other_count * other_mean
        ) / count

        combined_variance = (
            self.running_variance
            + other_var
            + mean_delta_squared * self.count * other_count / count
        )

        # Update our local variables to the newly combined statistics.
        self.running_mean = combined_mean
        self.running_variance = combined_variance
        self.count = count

    def serialize(self):
        return (
            self.running_mean.ravel().tolist()
            + self.running_variance.ravel().tolist()
            + [self.count]
        )

    def deserialize(self, other: np.ndarray):
        self.reset()
        n = np.prod(self.shape)

        other_mean = other[:n]
        other_var = other[n:-1]
        other_count = other[-1]
        self.running_mean = np.reshape(other_mean, self.shape)
        self.running_variance = np.reshape(other_var, self.shape)
        self.count = other_count

    def state_dict(self):
        return {
            "mean": self.running_mean.ravel().tolist(),
            "var": self.running_variance.ravel().tolist(),
            "shape": np.shape(self.running_mean),
            "count": self.count,
        }

    def load_state_dict(self, state: WelfordRunningStatStateDict):
        shape = state["shape"]
        self.count = state["count"]
        self.running_mean = np.asarray(state["mean"]).reshape(shape)
        self.running_variance = np.asarray(state["var"]).reshape(shape)
        print(
            f"LOADED RUNNING STATS FROM JSON | Mean: {self.running_mean} | Variance: {self.running_variance} | Count: {self.count}"
        )
