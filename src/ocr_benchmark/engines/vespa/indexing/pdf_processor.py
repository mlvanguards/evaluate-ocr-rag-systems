import logging
import os
import platform
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from pypdf import PdfReader
from torch import Tensor
from torch import device as torch_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ocr_benchmark.engines.vespa.datatypes import PDFData
from src.ocr_benchmark.engines.vespa.exceptions import PDFProcessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Class for processing PDFs and generating embeddings using ColQwen2 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v0.1",
        batch_size: int = 2,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the PDF processor with the specified model.

        Args:
            model_name: Name of the pretrained model to use
            batch_size: Batch size for processing images
            device: Optional specific device to use ('cuda', 'mps', 'cpu').
                   If None, will automatically select the optimal device.

        Raises:
            RuntimeError: If model loading fails
        """
        self.batch_size = batch_size
        self.device = self._get_optimal_device(device)

        try:
            logger.info(f"Loading model {model_name} on {self.device}")
            self.model = ColQwen2.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            self.processor = ColQwen2Processor.from_pretrained(model_name)
            self.model.eval()

            # Move model to device if not using device_map="auto"
            if self.device.type != "cuda":
                self.model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _get_optimal_device(
        self, requested_device: Optional[str] = None
    ) -> torch_device:
        """
        Determine the optimal device for model execution.

        Args:
            requested_device: Optional specific device to use

        Returns:
            torch.device: The optimal device for the current system
        """
        if requested_device:
            # If a specific device is requested, try to use it
            if requested_device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available, falling back to optimal device"
                )
            elif requested_device == "mps" and not (
                platform.system() == "Darwin" and torch.backends.mps.is_available()
            ):
                logger.warning(
                    "MPS requested but not available, falling back to optimal device"
                )
            else:
                return torch.device(requested_device)

        # Automatic device selection
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return torch.device("cuda")
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            try:
                # Test MPS allocation
                test_tensor = torch.zeros(1, device="mps")
                logger.info("Using MPS device")
                return torch.device("mps")
            except Exception as e:
                logger.warning(f"MPS device available but failed allocation test: {e}")
                logger.info("Falling back to CPU")
                return torch.device("cpu")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")

    def _handle_device_fallback(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], torch_device]:
        """
        Handle device fallback if the current device fails.

        Args:
            batch: The current batch of data

        Returns:
            Tuple of (processed batch, new device)

        Raises:
            PDFProcessingError: If processing fails on all devices
        """
        devices_to_try = []

        # Add fallback devices in order of preference
        if self.device.type != "cpu":
            devices_to_try.append("cpu")
        if self.device.type != "cuda" and torch.cuda.is_available():
            devices_to_try.append("cuda")

        for device_type in devices_to_try:
            try:
                new_device = torch.device(device_type)
                logger.warning(f"Attempting fallback to {device_type}")

                self.model.to(new_device)
                new_batch = {k: v.to(new_device) for k, v in batch.items()}

                # Test the new device with a forward pass
                with torch.no_grad():
                    _ = self.model(**new_batch)

                self.device = new_device
                return new_batch, new_device

            except Exception as e:
                logger.warning(f"Fallback to {device_type} failed: {e}")
                continue

        raise PDFProcessingError("Failed to process batch on all available devices")

    def process_pdf(self, pdf_metadata: List[Dict[str, str]]) -> List[PDFData]:
        """
        Process multiple PDFs and generate their embeddings.

        Args:
            pdf_metadata: List of dictionaries containing PDF metadata (must have 'url' and 'title')

        Returns:
            List of PDFData objects containing processed information

        Raises:
            PDFProcessingError: If processing any PDF fails
        """

        pdf_data: List[PDFData] = []

        for pdf in pdf_metadata:
            try:
                logger.info(f"Processing PDF: {pdf['title']}")
                images, texts = self.get_pdf_content(pdf["url"])
                embeddings = self.generate_embeddings(images)

                pdf_data.append(
                    PDFData(
                        url=pdf["url"],
                        title=pdf["title"],
                        images=images,
                        texts=texts,
                        embeddings=embeddings,
                    )
                )

            except Exception as e:
                logger.error(f"Failed to process PDF {pdf['title']}: {str(e)}")
                raise PDFProcessingError(f"Failed to process PDF: {str(e)}")

        return pdf_data

    def get_pdf_content(self, pdf_url: str) -> Tuple[List[Any], List[str]]:
        """
        Extract images and text content from PDF.

        Args:
            pdf_url: URL of the PDF to process

        Returns:
            Tuple containing lists of images and extracted text

        Raises:
            PDFProcessingError: If PDF processing fails
        """
        try:
            logger.info(f"Downloading PDF from {pdf_url}")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_file.read())

            try:
                reader = PdfReader(temp_path)
                page_texts = [page.extract_text() for page in reader.pages]

                logger.info("Converting PDF pages to images")
                images = convert_from_path(temp_path)

                if len(images) != len(page_texts):
                    raise PDFProcessingError(
                        "Mismatch between number of images and texts"
                    )

                return images, page_texts

            except PDFPageCountError as e:
                raise PDFProcessingError(f"Failed to process PDF pages: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_path)

        except requests.exceptions.RequestException as e:
            raise PDFProcessingError(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            raise PDFProcessingError(f"Failed to process PDF: {str(e)}")

    @torch.no_grad()
    def generate_embeddings(self, images: List[Any]) -> List[Tensor]:
        """
        Generate embeddings for a list of images.

        Args:
            images: List of PIL images to process

        Returns:
            List of tensor embeddings

        Raises:
            PDFProcessingError: If embedding generation fails
        """
        try:
            embeddings: List[Tensor] = []
            dataloader = DataLoader(
                images,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_images(x),
            )

            logger.info(f"Generating embeddings using device: {self.device}")
            for batch in tqdm(dataloader, desc="Processing batches"):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    batch_embeddings = self.model(**batch)
                    embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))

                except RuntimeError as e:
                    if any(err in str(e) for err in ["MPS", "CUDA", "device"]):
                        # Try fallback to another device
                        batch, new_device = self._handle_device_fallback(batch)
                        batch_embeddings = self.model(**batch)
                        embeddings.extend(
                            list(torch.unbind(batch_embeddings.to("cpu")))
                        )
                    else:
                        raise

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise PDFProcessingError(f"Embedding generation failed: {str(e)}")
