import asyncio

from src.ocr_benchmark.engines.vespa.indexing.run import PDFInput, run_indexing


async def index_documents():
    """Example function showing how to use run_indexing."""
    config_path = "/Users/vesaalexandru/Workspaces/cube/cube-publication/evaluate-ocr-rag-systems/src/ocr_benchmark/engines/vespa/vespa_config.yaml"

    pdfs = [
        PDFInput(
            title="Building a Resilient Strategy",
            url="https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf",
        )
    ]

    try:
        await run_indexing(config_path, pdfs)
    except Exception as e:
        print("error", e)
        raise


def main():
    """Entry point for the application."""
    asyncio.run(index_documents())


if __name__ == "__main__":
    main()
