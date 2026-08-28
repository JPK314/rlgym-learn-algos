import torch


class TensorCircularBuffer:
    def __init__(
        self,
        capacity: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ):
        self.storage: torch.Tensor = torch.empty(
            (2 * capacity, *shape),
            dtype=dtype,
            device=device,
            pin_memory=pin_memory and device.type == "cpu",
        )
        self.capacity: int = capacity
        self.length: int = 0
        self.write_pos: int = 0

    def append(self, t: torch.Tensor):
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

    def tensor(self) -> torch.Tensor:
        start = (self.write_pos - self.length) % self.capacity
        return self.storage[start : start + self.length]
