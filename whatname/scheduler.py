import math


class LinearSchedule:
    def __call__(self, t: int, T: int) -> float:
        return (t + 1) / T


class CosineSchedule:
    def __call__(self, t: int, T: int) -> float:
        # 0 -> small inf -> big
        x = (t + 1) / T
        return 1.0 - math.cos(math.pi * 0.5 * x) ** 2


class ConstantSchedule:
    def __init__(self, s=0.5): self.s = s
    def __call__(self, t: int, T: int) -> float: return self.s

