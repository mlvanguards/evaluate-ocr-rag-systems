import base64
import hashlib
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.ocr_benchmark.engines.vespa.indexing.pdf_processor import PDFData

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """Data class representing a processed PDF page for Vespa feed."""

    id: str
    url: str
    title: str
    page_number: int
    image: str  # Base64 encoded image
    text: str
    embedding: Dict[int, str]  # Binary vector embeddings


class VespaFeedError(Exception):
    """Custom exception for Vespa feed preparation errors."""

    pass


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""

    pass


class VespaFeedPreparator:
    """Class for preparing PDF data for Vespa feed."""

    def __init__(self, max_image_height: int = 800) -> None:
        """
        Initialize the Vespa feed preparator.

        Args:
            max_image_height: Maximum height for resized images
        """
        self.max_image_height = max_image_height

    def prepare_feed(self, pdf_data: List[PDFData]) -> List[PDFPage]:
        """
        Prepare PDF data for Vespa feed.

        Args:
            pdf_data: List of PDFData objects containing PDF data

        Returns:
            List of PDFPage objects ready for Vespa feed

        Raises:
            VespaFeedError: If feed preparation fails
        """

        try:
            vespa_feed: List[PDFPage] = []

            for pdf in pdf_data:
                logger.info(f"Processing PDF: {pdf.title}")
                pages = self._process_pdf_pages(pdf)
                vespa_feed.extend(pages)

            return vespa_feed

        except Exception as e:
            logger.error(f"Failed to prepare Vespa feed: {str(e)}")
            raise VespaFeedError(f"Feed preparation failed: {str(e)}")

    def _process_pdf_pages(self, pdf: PDFData) -> List[PDFPage]:
        """
        Process individual pages of a PDF.

        Args:
            pdf: PDFData object containing PDF data

        Returns:
            List of processed PDFPage objects

        Raises:
            VespaFeedError: If page processing fails
        """
        pages: List[PDFPage] = []

        try:
            for page_number, (text, embedding, image) in enumerate(
                zip(pdf.texts, pdf.embeddings, pdf.images)
            ):
                logger.debug(f"Processing page {page_number} of {pdf.title}")

                # Generate unique ID using SHA-256
                page_id = self._generate_page_id(pdf.url, page_number)

                # Process embedding vectors
                embedding_dict = self._process_embeddings(embedding)

                # Process and encode image
                processed_image = self._process_image(image)

                page = PDFPage(
                    id=page_id,
                    url=pdf.url,
                    title=pdf.title,
                    page_number=page_number,
                    image=processed_image,
                    text=text,
                    embedding=embedding_dict,
                )
                pages.append(page)

            return pages

        except Exception as e:
            logger.error(f"Failed to process PDF pages: {str(e)}")
            raise VespaFeedError(f"Page processing failed: {str(e)}")

    def _generate_page_id(self, url: str, page_number: int) -> str:
        """
        Generate a unique ID for a page using SHA-256.

        Args:
            url: PDF URL
            page_number: Page number

        Returns:
            Unique hash string for the page
        """
        content = f"{url}{page_number}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _process_embeddings(self, embedding: npt.NDArray[np.float32]) -> Dict[int, str]:
        """
        Process embedding vectors into binary format.

        Args:
            embedding: Numpy array of embedding vectors

        Returns:
            Dictionary mapping indices to binary vector strings

        Raises:
            VespaFeedError: If embedding processing fails
        """
        try:
            embedding_dict: Dict[int, str] = {}

            for idx, patch_embedding in enumerate(embedding):
                binary_vector = self._convert_to_binary_vector(patch_embedding)
                embedding_dict[idx] = binary_vector

            return embedding_dict

        except Exception as e:
            logger.error(f"Failed to process embeddings: {str(e)}")
            raise VespaFeedError(f"Embedding processing failed: {str(e)}")

    @staticmethod
    def _convert_to_binary_vector(patch_embedding: npt.NDArray[np.float32]) -> str:
        """
        Convert embedding vector to binary format.

        Args:
            patch_embedding: Single embedding vector

        Returns:
            Hexadecimal string representation of binary vector
        """
        binary = np.packbits(np.where(patch_embedding > 0, 1, 0))
        return binary.astype(np.int8).tobytes().hex()

    def _process_image(self, image: Image.Image) -> str:
        """
        Process and encode image in base64.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string of the processed image

        Raises:
            ImageProcessingError: If image processing fails
        """
        try:
            # Resize image if needed
            resized_image = self._resize_image(image)

            # Convert to base64
            return self._encode_image_base64(resized_image)

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise ImageProcessingError(f"Image processing failed: {str(e)}")

    def _resize_image(
        self, image: Image.Image, target_width: Optional[int] = 640
    ) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image: PIL Image object
            target_width: Target width for the resized image

        Returns:
            Resized PIL Image object
        """
        width, height = image.size

        if height > self.max_image_height:
            ratio = self.max_image_height / height
            new_width = int(width * ratio)
            return image.resize(
                (new_width, self.max_image_height), Image.Resampling.LANCZOS
            )

        if target_width and width > target_width:
            ratio = target_width / width
            new_height = int(height * ratio)
            return image.resize((target_width, new_height), Image.Resampling.LANCZOS)

        return image

    @staticmethod
    def _encode_image_base64(image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string

        Raises:
            ImageProcessingError: If encoding fails
        """
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            raise ImageProcessingError(f"Image encoding failed: {str(e)}")
