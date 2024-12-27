import base64
from io import BytesIO

import numpy as np


def prepare_vespa_feed(pdf_data):
    vespa_feed = []
    for pdf in pdf_data:
        for page_number, (text, embedding, image) in enumerate(
            zip(pdf["texts"], pdf["embeddings"], pdf["images"])
        ):
            embedding_dict = {}
            for idx, patch_embedding in enumerate(embedding):
                binary_vector = (
                    np.packbits(np.where(patch_embedding > 0, 1, 0))
                    .astype(np.int8)
                    .tobytes()
                    .hex()
                )
                embedding_dict[idx] = binary_vector

            page = {
                "id": hash(pdf["url"] + str(page_number)),
                "url": pdf["url"],
                "title": pdf["title"],
                "page_number": page_number,
                "image": get_base64_image(resize_image(image, 640)),
                "text": text,
                "embedding": embedding_dict,
            }
            vespa_feed.append(page)
    return vespa_feed


def resize_image(image, max_height=800):
    width, height = image.size
    if height > max_height:
        ratio = max_height / height
        return image.resize((int(width * ratio), int(height * ratio)))
    return image


def get_base64_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return str(base64.b64encode(buffered.getvalue()), "utf-8")
