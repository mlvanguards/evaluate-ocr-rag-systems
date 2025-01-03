from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class EmbeddingType(Enum):
    COLPALI = "ColPali"
    TRADITIONAL = "Traditional"


@dataclass
class AnalysisConfig:
    """Configuration parameters for PDF analysis"""

    visual_threshold: int = 15
    text_density_threshold: float = 0.25
    layout_threshold: int = 100
    min_image_size: int = 1000
    table_row_threshold: int = 5
    table_weight: float = 0.3


@dataclass
class AnalysisResult:
    """Structured container for analysis results"""

    score: float
    details: Dict[str, Any]
    confidence: float = 1.0
