import numpy as np
import torch


class TensorCircularBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        pin_memory: bool,
    ):
        assert capacity > 0, "Capacity must be a positive integer"
        self.storage: torch.Tensor | None = None
        self.capacity: int = capacity
        self.device: torch.device = device
        self.pin_memory: bool = pin_memory
        self.length: int = 0
        self.write_pos: int = 0

    def append(self, t: torch.Tensor):
        if self.storage is None:
            self.storage = torch.empty(
                (2 * self.capacity, *tuple(t.shape[1:])),
                dtype=t.dtype,
                device=self.device,
                pin_memory=self.pin_memory and self.device.type == "cpu",
            )
        n = len(t)

        if n == 0:
            return

        if n >= self.capacity:
            source = t[-self.capacity :]
            _ = self.storage[: self.capacity].copy_(source)
            _ = self.storage[self.capacity :].copy_(source)
            self.length = self.capacity
            self.write_pos = 0
            return

        # first_length is the length of t that can fit up to the end of the buffer (before the mirrored copy)
        # second_length is the length of t after that until the end of t
        first_length = min(n, self.capacity - self.write_pos)
        second_length = n - first_length

        first_source = t[:first_length]
        first_start = self.write_pos
        first_stop = first_start + first_length

        _ = self.storage[first_start:first_stop].copy_(first_source)
        _ = self.storage[
            first_start + self.capacity : first_stop + self.capacity
        ].copy_(first_source)

        if second_length:
            second_source = t[first_length:]

            _ = self.storage[:second_length].copy_(second_source)
            _ = self.storage[self.capacity : self.capacity + second_length].copy_(
                second_source
            )

        self.write_pos = (self.write_pos + n) % self.capacity
        self.length = min(self.length + n, self.capacity)

    def tensor(self) -> torch.Tensor | None:
        if self.storage is None:
            return None
        start = (self.write_pos - self.length) % self.capacity
        return self.storage[start : start + self.length]


class NumpyCircularBuffer:
    def __init__(
        self,
        capacity: int,
    ):
        assert capacity > 0, "Capacity must be a positive integer"
        self.storage: np.ndarray | None = None
        self.capacity: int = capacity
        self.length: int = 0
        self.write_pos: int = 0

    def append(self, arr: np.ndarray):
        if self.storage is None:
            self.storage = np.empty(
                (2 * self.capacity, *arr.shape[1:]),
                dtype=arr.dtype,
            )
        n = len(arr)

        if n == 0:
            return

        if n >= self.capacity:
            source = arr[-self.capacity :]
            self.storage[: self.capacity] = source
            self.storage[self.capacity :] = source
            self.length = self.capacity
            self.write_pos = 0
            return

        # first_length is the length of t that can fit up to the end of the buffer (before the mirrored copy)
        # second_length is the length of t after that until the end of t
        first_length = min(n, self.capacity - self.write_pos)
        second_length = n - first_length

        first_source = arr[:first_length]
        first_start = self.write_pos
        first_stop = first_start + first_length

        self.storage[first_start:first_stop] = first_source
        self.storage[first_start + self.capacity : first_stop + self.capacity] = (
            first_source
        )

        if second_length:
            second_source = arr[first_length:]

            self.storage[:second_length] = second_source
            self.storage[self.capacity : self.capacity + second_length] = second_source

        self.write_pos = (self.write_pos + n) % self.capacity
        self.length = min(self.length + n, self.capacity)

    def array(self) -> np.ndarray | None:
        if self.storage is None:
            return None
        start = (self.write_pos - self.length) % self.capacity
        return self.storage[start : start + self.length]
