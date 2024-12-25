"""Base class for OCR engines following the Interface Segregation Principle."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from PIL import Image


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the OCR engine."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the OCR engine with required models and configurations."""
        pass

    @abstractmethod
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a single image and return the extracted text and metadata.

        Args:
            image: PIL Image object to process

        Returns:
            Dictionary containing:
                - text: extracted text
                - confidence: confidence scores if available
                - boxes: bounding boxes if available
                - tables: detected tables if available
        """
        pass

    @abstractmethod
    def process_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Process multiple images and return results for each.

        Args:
            images: List of PIL Image objects

        Returns:
            List of dictionaries containing results for each image
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Override if needed."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
