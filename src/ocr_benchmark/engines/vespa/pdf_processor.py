from io import BytesIO

import requests
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from pypdf import PdfReader
from torch.utils.data import DataLoader
from tqdm import tqdm


class PDFProcessor:
    def __init__(self, model_name="vidore/colqwen2-v0.1"):
        self.model = ColQwen2.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.model.eval()

    def download_pdf(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")

    def get_pdf_content(self, pdf_url):
        pdf_file = self.download_pdf(pdf_url)
        temp_file = "temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(pdf_file.read())

        reader = PdfReader(temp_file)
        page_texts = [page.extract_text() for page in reader.pages]
        images = convert_from_path(temp_file)
        assert len(images) == len(page_texts)
        return images, page_texts

    def process_pdf(self, pdf_metadata):
        pdf_data = []
        for pdf in pdf_metadata:
            images, texts = self.get_pdf_content(pdf["url"])
            embeddings = self.generate_embeddings(images)
            pdf_data.append(
                {
                    "url": pdf["url"],
                    "title": pdf["title"],
                    "images": images,
                    "texts": texts,
                    "embeddings": embeddings,
                }
            )
        return pdf_data

    def generate_embeddings(self, images):
        embeddings = []
        dataloader = DataLoader(
            images,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )

        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                batch_embeddings = self.model(**batch)
                embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
        return embeddings
