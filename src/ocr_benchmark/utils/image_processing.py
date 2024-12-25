"""Utility functions for image processing and manipulation."""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


@dataclass
class ImagePreprocessingConfig:
    """Configuration for image preprocessing."""

    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    contrast_factor: float = 1.0
    brightness_factor: float = 1.0
    sharpen_factor: float = 1.0
    denoise: bool = False
    deskew: bool = False
    binarize: bool = False


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure image is in RGB format.

    Args:
        image: Input PIL Image

    Returns:
        PIL Image in RGB format
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_image(
    image: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Resize image while optionally maintaining aspect ratio.

    Args:
        image: Input PIL Image
        width: Target width in pixels
        height: Target height in pixels
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized PIL Image
    """
    if not width and not height:
        return image

    orig_width, orig_height = image.size

    if maintain_aspect:
        if width and height:
            # Use the dimension that results in a smaller image
            width_ratio = width / orig_width
            height_ratio = height / orig_height
            ratio = min(width_ratio, height_ratio)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
        elif width:
            ratio = width / orig_width
            new_width = width
            new_height = int(orig_height * ratio)
        else:
            ratio = height / orig_height
            new_width = int(orig_width * ratio)
            new_height = height
    else:
        new_width = width if width else orig_width
        new_height = height if height else orig_height

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def enhance_image(
    image: Image.Image,
    contrast: float = 1.0,
    brightness: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    """
    Enhance image using contrast, brightness, and sharpness adjustments.

    Args:
        image: Input PIL Image
        contrast: Contrast enhancement factor
        brightness: Brightness enhancement factor
        sharpness: Sharpness enhancement factor

    Returns:
        Enhanced PIL Image
    """
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


def denoise_image(image: Image.Image) -> Image.Image:
    """
    Apply denoising to image.

    Args:
        image: Input PIL Image

    Returns:
        Denoised PIL Image
    """
    # Convert to numpy array for OpenCV processing
    img_array = np.array(image)
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    return Image.fromarray(denoised)


def deskew_image(image: Image.Image) -> Image.Image:
    """
    Deskew image by detecting and correcting rotation.

    Args:
        image: Input PIL Image

    Returns:
        Deskewed PIL Image
    """
    # Convert to numpy array and grayscale
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        # Calculate the most common angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)

        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:  # Only rotate if angle is significant
                return image.rotate(
                    median_angle, expand=True, fillcolor=(255, 255, 255)
                )

    return image


def binarize_image(image: Image.Image) -> Image.Image:
    """
    Convert image to binary (black and white) using adaptive thresholding.

    Args:
        image: Input PIL Image

    Returns:
        Binarized PIL Image
    """
    # Convert to grayscale
    gray = ImageOps.grayscale(image)

    # Convert to numpy array for OpenCV processing
    img_array = np.array(gray)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img_array,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2,  # Constant subtracted from mean
    )

    return Image.fromarray(binary)


def preprocess_image(
    image: Image.Image, config: ImagePreprocessingConfig
) -> Image.Image:
    """
    Apply a series of preprocessing steps to an image.

    Args:
        image: Input PIL Image
        config: Preprocessing configuration

    Returns:
        Preprocessed PIL Image
    """
    try:
        # Ensure RGB format
        image = ensure_rgb(image)

        # Resize if needed
        if config.resize_width or config.resize_height:
            image = resize_image(image, config.resize_width, config.resize_height)

        # Apply enhancements
        if any(
            factor != 1.0
            for factor in [
                config.contrast_factor,
                config.brightness_factor,
                config.sharpen_factor,
            ]
        ):
            image = enhance_image(
                image,
                config.contrast_factor,
                config.brightness_factor,
                config.sharpen_factor,
            )

        # Apply denoising
        if config.denoise:
            image = denoise_image(image)

        # Apply deskewing
        if config.deskew:
            image = deskew_image(image)

        # Apply binarization
        if config.binarize:
            image = binarize_image(image)

        return image

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise
