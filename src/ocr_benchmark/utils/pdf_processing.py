"""Utility functions for PDF processing and conversion."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pdf2image
from pdf2image.exceptions import (
    PDFPageCountError,
    PDFSyntaxError,
)
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PDFConversionSettings:
    """Settings for PDF to image conversion."""

    dpi: int = 300
    grayscale: bool = False
    use_cropbox: bool = False
    strict: bool = False
    thread_count: int = 4
    raise_on_error: bool = True


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""

    pass


def validate_pdf(pdf_path: Path) -> Tuple[bool, str]:
    """
    Validate PDF file existence and readability.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if not pdf_path.exists():
            return False, f"File not found: {pdf_path}"

        if pdf_path.stat().st_size == 0:
            return False, f"File is empty: {pdf_path}"

        # Try converting first page to validate format
        pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1)
        return True, "Valid PDF file"

    except PDFSyntaxError:
        return False, f"Invalid PDF format or corrupted file: {pdf_path}"
    except PDFPageCountError:
        return False, f"Error determining page count: {pdf_path}"
    except Exception as e:
        return False, f"Error validating PDF: {str(e)}"


def pdf_to_images(
    pdf_path: Path, settings: Optional[PDFConversionSettings] = None
) -> List[Image.Image]:
    """
    Convert PDF file to a list of PIL Images.

    Args:
        pdf_path: Path to PDF file
        settings: Optional conversion settings

    Returns:
        List of PIL Images, one per page

    Raises:
        PDFProcessingError: If conversion fails and raise_on_error is True
    """
    if settings is None:
        settings = PDFConversionSettings()

    try:
        # Validate PDF first
        is_valid, message = validate_pdf(pdf_path)
        if not is_valid and settings.raise_on_error:
            raise PDFProcessingError(message)

        # Create temporary directory for conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert PDF to images
                images = pdf2image.convert_from_path(
                    str(pdf_path),
                    dpi=settings.dpi,
                    grayscale=settings.grayscale,
                    use_cropbox=settings.use_cropbox,
                    strict=settings.strict,
                    thread_count=settings.thread_count,
                    output_folder=temp_dir,
                )

                logger.info(f"Successfully converted PDF with {len(images)} pages")
                return images

            except Exception as e:
                error_msg = f"Error converting PDF to images: {str(e)}"
                if settings.raise_on_error:
                    raise PDFProcessingError(error_msg)
                logger.error(error_msg)
                return []

    except Exception as e:
        error_msg = f"PDF processing error: {str(e)}"
        if settings.raise_on_error:
            raise PDFProcessingError(error_msg)
        logger.error(error_msg)
        return []


def estimate_pdf_size(pdf_path: Path) -> Tuple[int, str]:
    """
    Estimate memory requirements for PDF conversion.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (size_in_bytes, human_readable_size)
    """
    try:
        # Convert first page to estimate size per page
        sample = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1)[
            0
        ]
        width, height = sample.size
        channels = len(sample.getbands())

        # Get total page count
        info = pdf2image.pdfinfo_from_path(str(pdf_path))
        total_pages = info["Pages"]

        # Estimate total memory requirement
        bytes_per_page = width * height * channels
        total_bytes = bytes_per_page * total_pages

        # Convert to human readable
        for unit in ["B", "KB", "MB", "GB"]:
            if total_bytes < 1024:
                return total_bytes, f"{total_bytes:.1f}{unit}"
            total_bytes /= 1024

        return total_bytes, f"{total_bytes:.1f}GB"

    except Exception as e:
        logger.error(f"Error estimating PDF size: {str(e)}")
        return 0, "Unknown"
