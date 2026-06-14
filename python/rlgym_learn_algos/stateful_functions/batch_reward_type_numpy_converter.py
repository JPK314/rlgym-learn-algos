# pyright: reportUnusedParameter=false

from typing import Any, Generic

import numpy as np
from numpy import dtype, ndarray
from rlgym.api import RewardType
from typing_extensions import override


class BatchRewardTypeNumpyConverter(Generic[RewardType]):
    def __init__(self):
        self.dtype: dtype

    def as_numpy(self, rewards: list[RewardType]) -> np.ndarray:
        """
        :param rewards: A list of RewardType to be converted
        :return: A Numpy array of shape (n,) parallel to the input list, with dtype self.dtype
        """
        raise NotImplementedError

    def set_dtype(self, dtype: dtype):
        self.dtype = dtype


class BatchRewardTypeSimpleNumpyConverter(BatchRewardTypeNumpyConverter[Any]):
    @override
    def as_numpy(self, rewards: list[Any]) -> ndarray:
        return np.array(rewards, dtype=self.dtype)
