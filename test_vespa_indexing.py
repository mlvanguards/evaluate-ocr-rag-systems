import asyncio

from src.ocr_benchmark.engines.vespa.indexing.pdf_processor import PDFProcessor
from src.ocr_benchmark.engines.vespa.indexing.prepare_feed import VespaFeedPreparator
from src.ocr_benchmark.engines.vespa.indexing.run import PDFInput, run_indexing


async def index_documents():
    """Example function showing how to use run_indexing."""
    config_path = "/Users/vesaalexandru/Workspaces/cube/cube-publication/evaluate-ocr-rag-systems/src/ocr_benchmark/engines/vespa/vespa_config.yaml"

    pdfs = [
        PDFInput(
            title="Building a Resilient Strategy",
            url="https://static.conocophillips.com/files/resources/24-0976-sustainability-highlights_nature.pdf",
        )
    ]

    # pdfs = [
    #     PDFInput(
    #         title="Building a Resilient Strategy",
    #         url="https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf",
    #     )
    # ]

    # Explicitly create the processors
    pdf_processor = PDFProcessor()
    feed_preparator = VespaFeedPreparator()

    try:
        await run_indexing(
            config_path=config_path,
            pdfs=pdfs,
            pdf_processor=pdf_processor,
            feed_preparator=feed_preparator,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


def main():
    """Entry point for the application."""
    asyncio.run(index_documents())


if __name__ == "__main__":
    main()
