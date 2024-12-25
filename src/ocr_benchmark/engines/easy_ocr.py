"""EasyOCR engine implementation."""

from typing import Any, Dict, List, Optional

import easyocr
import numpy as np
from PIL import Image

from src.ocr_benchmark.base import OCREngine
from src.ocr_benchmark.engines.config import EasyOCRConfig
from src.ocr_benchmark.utils.image_processing import ensure_rgb


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""

    def __init__(self, config: Optional[EasyOCRConfig] = None):
        """
        Initialize EasyOCREngine.

        Args:
            config: Optional configuration for EasyOCR
        """
        self._config = config or EasyOCRConfig()
        self._initialized = False
        self._reader = None

    @property
    def name(self) -> str:
        return "EasyOCR"

    def initialize(self) -> None:
        """Initialize EasyOCR with configuration."""
        if not self._initialized:
            self._reader = easyocr.Reader(
                lang_list=self._config.languages,
                gpu=self._config.gpu,
                model_storage_directory=self._config.model_storage_directory,
                download_enabled=self._config.download_enabled,
            )
            self._initialized = True

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a single image using EasyOCR.

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

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Process image with EasyOCR
        results = self._reader.readtext(image_np)

        # Extract text and confidence scores
        boxes = []
        full_text = []
        confidences = []

        for bbox, text, conf in results:
            full_text.append(text)
            confidences.append(conf)

            # Convert bbox points to (x1, y1, x2, y2) format
            x1 = min(point[0] for point in bbox)
            y1 = min(point[1] for point in bbox)
            x2 = max(point[0] for point in bbox)
            y2 = max(point[1] for point in bbox)

            boxes.append(
                {
                    "text": text,
                    "conf": conf,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

        # Calculate mean confidence
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "text": " ".join(full_text),
            "confidence": mean_confidence,
            "boxes": boxes,
        }

    def process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images and return results for each."""
        return [self.process_image(image) for image in images]


if __name__ == "__main__":
    import os

    from PIL import Image, ImageDraw, ImageFont

    # Create a test image
    img = Image.new("RGB", (800, 200), color="white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 60)
    except:
        font = ImageFont.load_default()

    d.text((50, 50), "Hello, EasyOCR Test!", fill="black", font=font)

    # Save test image
    test_image_path = "test_easyocr.png"
    img.save(test_image_path)

    try:
        # Initialize engine
        engine = EasyOCREngine(EasyOCRConfig(languages=["en"]))

        # Process test image
        with Image.open(test_image_path) as test_img:
            result = engine.process_image(test_img)

        # Print results
        print("\nEasyOCR Test Results:")
        print(f"Detected Text: {result['text']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("Detected Boxes:", len(result["boxes"]))

        for box in result["boxes"]:
            print(f"- Text: {box['text']}, Confidence: {box['conf']:.2f}")
            print(f"  Box coordinates: {box['box']}")

    except Exception as e:
        print(f"Error during test: {str(e)}")

    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
