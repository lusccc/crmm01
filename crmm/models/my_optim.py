from typing import Optional, Callable

from torch.optim import AdamW


class DBNOptmi(AdamW):
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        return super().step(closure)


class NormalOptmi(AdamW):
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        return super().step(closure)