"""DocTR engine implementation for OCR benchmark."""

from typing import Any, Dict, List, Optional

import numpy as np
from doctr.models import ocr_predictor
from PIL import Image

from src.ocr_benchmark.base import OCREngine
from src.ocr_benchmark.engines.config import DocTRConfig
from src.ocr_benchmark.utils.image_processing import ensure_rgb


class DocTREngineError(Exception):
    """Custom exception for DocTR engine errors."""

    pass


class DocTREngine(OCREngine):
    """DocTR engine implementation."""

    def __init__(self, config: Optional[DocTRConfig] = None):
        """Initialize DocTREngine."""
        self._config = config or DocTRConfig()
        self._initialized = False
        self._model = None

    @property
    def name(self) -> str:
        return "DocTR"

    def initialize(self) -> None:
        """Initialize DocTR with configuration."""
        try:
            if not self._initialized:
                self._model = ocr_predictor(
                    pretrained=self._config.pretrained,
                    assume_straight_pages=self._config.assume_straight_pages,
                    straighten_pages=self._config.straighten_pages,
                )
                self._initialized = True
        except Exception as e:
            raise DocTREngineError(f"Failed to initialize DocTR: {str(e)}")

    def _extract_text_and_boxes(
        self, result, image_size
    ) -> tuple[List[str], List[float], List[Dict]]:
        """Extract text, confidences, and bounding boxes from DocTR result."""
        boxes = []
        confidences = []
        text_list = []
        width, height = image_size

        try:
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            text = word.value

                            # In newer DocTR versions, confidence might not be available
                            conf = getattr(word, "confidence", 0.0)

                            if text.strip():
                                text_list.append(text)
                                confidences.append(conf)

                                # Get geometry - it's stored as relative coordinates
                                # DocTR uses (top, left, bottom, right) format
                                bbox = word.geometry
                                if isinstance(bbox, tuple) and len(bbox) == 2:
                                    # Handle case where geometry is ((x,y), (x,y))
                                    (x1, y1), (x2, y2) = bbox
                                else:
                                    # Handle case where geometry is [y1, x1, y2, x2]
                                    y1, x1, y2, x2 = bbox

                                # Convert to absolute coordinates
                                abs_box = (
                                    int(x1 * width),  # left
                                    int(y1 * height),  # top
                                    int(x2 * width),  # right
                                    int(y2 * height),  # bottom
                                )

                                boxes.append(
                                    {"text": text, "conf": float(conf), "box": abs_box}
                                )

        except Exception as e:
            raise DocTREngineError(f"Error extracting text and boxes: {str(e)}")

        return text_list, confidences, boxes

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image using DocTR."""
        try:
            if not self._initialized:
                self.initialize()

            # Ensure image is in RGB format
            image = ensure_rgb(image)

            # Convert PIL Image to numpy array
            image_np = np.array(image)

            # Process with DocTR
            result = self._model([image_np])

            # Extract information
            text_list, confidences, boxes = self._extract_text_and_boxes(
                result, image.size
            )

            # Calculate mean confidence
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                "text": " ".join(text_list),
                "confidence": mean_confidence,
                "boxes": boxes,
            }

        except Exception as e:
            raise DocTREngineError(f"Error processing image with DocTR: {str(e)}")

    def process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images and return results for each."""
        results = []
        for idx, image in enumerate(images):
            try:
                result = self.process_image(image)
                results.append(result)
            except DocTREngineError as e:
                print(f"Warning: Failed to process image {idx}: {str(e)}")
                results.append(
                    {"text": "", "confidence": 0.0, "boxes": [], "error": str(e)}
                )
        return results


if __name__ == "__main__":
    import logging
    import os

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def create_test_image(text: str, size: tuple = (800, 200)) -> Image.Image:
        """Create a test image with given text."""
        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)

        # Try to get a larger font size for better OCR
        font_size = min(size) // 4  # Adjust font size based on image size
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            try:
                # Try system font locations
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", font_size
                )
            except OSError:
                font = ImageFont.load_default()
                logger.warning("Standard fonts not found, using default font")

        # Calculate text position to center it
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        # Draw text with black color
        draw.text((x, y), text, fill="black", font=font)
        return img


if __name__ == "__main__":
    """Run a series of tests on the DocTR engine."""
    # Test cases
    test_cases = [
        {"name": "Basic Text", "text": "Hello DocTR Test", "size": (800, 200)},
        {"name": "Numbers", "text": "12345 67890", "size": (800, 200)},
        {"name": "Mixed Content", "text": "Test ABC 123", "size": (800, 200)},
    ]

    engine = DocTREngine(DocTRConfig(pretrained=True))

    for test in test_cases:
        logger.info(f"\nRunning test: {test['name']}")

        # Create test image
        test_image = create_test_image(test["text"], test["size"])
        test_path = f"doctr_test_{test['name'].lower().replace(' ', '_')}.png"
        test_image.save(test_path)

        try:
            # Process image
            result = engine.process_image(test_image)

            # Log results
            logger.info(f"Expected Text: {test['text']}")
            logger.info(f"Detected Text: {result['text']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"Number of boxes: {len(result['boxes'])}")

            # Log individual box details
            for box in result["boxes"]:
                logger.info(f"- Text: {box['text']}")
                logger.info(f"  Confidence: {box['conf']:.2f}")
                logger.info(f"  Box: {box['box']}")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

        finally:
            # Cleanup
            if os.path.exists(test_path):
                os.remove(test_path)
