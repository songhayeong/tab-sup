from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class DatasetSpec:
    cat_features: List[str]
    num_features: List[str]
    label: str
    # continuous binning
    binning: Optional[Dict[str, Dict]] = None


@dataclass
class BuildParams:
    smoothing_alpha: float = 1.0
    weight_clip: Tuple[float, float] = (0.0, 50.0)
    dt: float = 0.05    # Euler step
    steps: int = 100    # Forward steps
    temperature: float = 1.0    # guide temp