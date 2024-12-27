import base64
from io import BytesIO

import numpy as np
import requests
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from pypdf import PdfReader
from torch.utils.data import DataLoader
from tqdm import tqdm
from vespa.deployment import VespaCloud
from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    FirstPhaseRanking,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)


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


class VespaSetup:
    def __init__(self, app_name):
        self.app_name = app_name
        self.schema = self._create_schema()
        self.app_package = ApplicationPackage(name=app_name, schema=[self.schema])

    def _create_schema(self):
        schema = Schema(
            name="pdf_page",
            document=Document(
                fields=[
                    Field(
                        name="id",
                        type="string",
                        indexing=["summary", "index"],
                        match=["word"],
                    ),
                    Field(name="url", type="string", indexing=["summary", "index"]),
                    Field(
                        name="title",
                        type="string",
                        indexing=["summary", "index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="page_number",
                        type="int",
                        indexing=["summary", "attribute"],
                    ),
                    Field(name="image", type="raw", indexing=["summary"]),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="embedding",
                        type="tensor<int8>(patch{}, v[16])",
                        indexing=["attribute", "index"],
                        ann=HNSW(
                            distance_metric="hamming",
                            max_links_per_node=32,
                            neighbors_to_explore_at_insert=400,
                        ),
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "text"])],
        )
        self._add_rank_profiles(schema)
        return schema

    def _add_rank_profiles(self, schema):
        default_profile = RankProfile(
            name="default",
            inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
            functions=[
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)) , v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(name="bm25_score", expression="bm25(title) + bm25(text)"),
            ],
            first_phase=FirstPhaseRanking(expression="bm25_score"),
            second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=100),
        )
        schema.add_rank_profile(default_profile)


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


async def deploy_and_feed(vespa_feed):
    vespa_setup = VespaSetup("test")

    vespa_cloud = VespaCloud(
        tenant="cube-digital",
        application="test",
        application_package=vespa_setup.app_package,
    )

    app = vespa_cloud.deploy()

    async with app.asyncio(connections=1, timeout=180) as session:
        for page in tqdm(vespa_feed):
            response = await session.feed_data_point(
                data_id=page["id"], fields=page, schema="pdf_page"
            )
            if not response.is_successful():
                print(response.json())
    return app


async def main():
    # Example usage
    sample_pdfs = [
        {
            "title": "Building a Resilient Strategy for the Energy Transition",
            "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf",
        }
    ]

    processor = PDFProcessor()
    pdf_data = processor.process_pdf(sample_pdfs)
    vespa_feed = prepare_vespa_feed(pdf_data)

    # Deploy to Vespa Cloud (requires configuration)
    await deploy_and_feed(vespa_feed=vespa_feed)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    ()
