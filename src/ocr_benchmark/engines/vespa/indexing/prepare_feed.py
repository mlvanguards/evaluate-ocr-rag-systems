import base64
import hashlib
import logging
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image
from torch import Tensor

from src.ocr_benchmark.engines.vespa.indexing.pdf_processor import PDFData

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class VespaFeedPreparator:
    def __init__(self, max_image_height: int = 800) -> None:
        self.max_image_height = max_image_height

    def prepare_feed(self, pdf_data: List[PDFData]) -> List[Dict]:
        try:
            vespa_feed = []
            for pdf in pdf_data:
                logger.info(f"Processing PDF: {pdf.title}")
                for page_number, (text, embedding, image) in enumerate(
                    zip(pdf.texts, pdf.embeddings, pdf.images)
                ):
                    page_id = self._generate_page_id(pdf.url, page_number)
                    processed_image = self._process_image(image)
                    embedding_dict = self._process_embeddings(embedding)

                    doc = {
                        "fields": {
                            "id": page_id,
                            "url": pdf.url,
                            "title": pdf.title,
                            "page_number": page_number,
                            "image": processed_image,
                            "text": text,
                            "embedding": {
                                "blocks": self._convert_to_patch_blocks(embedding_dict)
                            },
                        }
                    }
                    vespa_feed.append(doc)
            return vespa_feed
        except Exception as e:
            logger.error(f"Failed to prepare feed: {str(e)}")
            raise

    def _convert_to_patch_blocks(self, embedding_dict: Dict[int, str]) -> List[Dict]:
        return [
            {"address": {"patch": patch_idx}, "values": vector}
            for patch_idx, vector in embedding_dict.items()
            if vector != "0" * 32
        ]

    def _generate_page_id(self, url: str, page_number: int) -> str:
        content = f"{url}{page_number}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _process_embeddings(self, embedding: Tensor) -> Dict[int, str]:
        embedding_dict = {}
        embedding_float = embedding.detach().cpu().float()
        embedding_np = embedding_float.numpy()
        for idx, patch_embedding in enumerate(embedding_np):
            binary_vector = self._convert_to_binary_vector(patch_embedding)
            embedding_dict[idx] = binary_vector
        return embedding_dict

    @staticmethod
    def _convert_to_binary_vector(patch_embedding: npt.NDArray[np.float32]) -> str:
        binary = np.packbits(np.where(patch_embedding > 0, 1, 0))
        return binary.astype(np.int8).tobytes().hex()

    def _process_image(self, image: Image.Image) -> str:
        resized_image = self._resize_image(image)
        return self._encode_image_base64(resized_image)

    def _resize_image(
        self, image: Image.Image, target_width: Optional[int] = 640
    ) -> Image.Image:
        width, height = image.size
        if height > self.max_image_height:
            ratio = self.max_image_height / height
            new_width = int(width * ratio)
            return image.resize((new_width, self.max_image_height), Image.LANCZOS)
        if target_width and width > target_width:
            ratio = target_width / width
            new_height = int(height * ratio)
            return image.resize((target_width, new_height), Image.LANCZOS)
        return image

    @staticmethod
    def _encode_image_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
