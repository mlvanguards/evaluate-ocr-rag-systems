"""Configuration classes for OCR engines."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TesseractConfig:
    """Configuration for Tesseract OCR."""

    tessdata_path: Optional[str] = None
    language: str = "eng"
    psm: int = 3  # Page segmentation mode
    oem: int = 3  # OCR Engine mode


@dataclass
class EasyOCRConfig:
    """Configuration for EasyOCR."""

    languages: List[str] = ("en",)
    gpu: bool = False
    model_storage_directory: Optional[str] = None
    download_enabled: bool = True


@dataclass
class PaddleOCRConfig:
    """Configuration for PaddleOCR."""

    use_angle_cls: bool = True
    lang: str = "en"
    use_gpu: bool = False
    show_log: bool = False


@dataclass
class DocTRConfig:
    """Configuration for DocTR."""

    pretrained: bool = True
    assume_straight_pages: bool = True
    straighten_pages: bool = True
