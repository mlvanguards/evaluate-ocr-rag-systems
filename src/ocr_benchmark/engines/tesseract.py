"""Tesseract OCR engine implementation."""

from typing import Any, Dict, List, Optional

import pytesseract
from PIL import Image

from src.ocr_benchmark.base import OCREngine
from src.ocr_benchmark.engines.config import TesseractConfig
from src.ocr_benchmark.utils.image_processing import ensure_rgb


class TesseractEngine(OCREngine):
    """Tesseract OCR engine implementation."""

    def __init__(self, config: Optional[TesseractConfig] = None):
        """
        Initialize TesseractEngine.

        Args:
            config: Optional configuration for Tesseract
        """
        self._config = config or TesseractConfig()
        self._initialized = False

    @property
    def name(self) -> str:
        return "Tesseract"

    def initialize(self) -> None:
        """Initialize Tesseract with configuration."""
        if not self._initialized:
            if self._config.tessdata_path:
                pytesseract.pytesseract.tesseract_cmd = self._config.tessdata_path
            self._initialized = True

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a single image using Tesseract.

        Args:
            image: PIL Image to process

        Returns:
            Dictionary containing:
                - text: extracted text
                - confidence: mean confidence score
                - boxes: detected text boxes
        """
        if not self._initialized:
            self.initialize()

        # Ensure image is in RGB format
        image = ensure_rgb(image)

        # Get text and data
        text = pytesseract.image_to_string(image)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Calculate mean confidence for non-empty text
        confidences = [
            conf
            for conf, text in zip(data["conf"], data["text"])
            if text.strip() and conf != -1
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Get bounding boxes
        boxes = []
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                boxes.append(
                    {
                        "text": data["text"][i],
                        "conf": data["conf"][i],
                        "box": (
                            data["left"][i],
                            data["top"][i],
                            data["left"][i] + data["width"][i],
                            data["top"][i] + data["height"][i],
                        ),
                    }
                )

        return {"text": text, "confidence": mean_confidence, "boxes": boxes}

    def process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images and return results for each."""
        return [self.process_image(image) for image in images]
