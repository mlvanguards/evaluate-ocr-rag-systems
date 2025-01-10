import asyncio
import logging
from typing import Dict, List

from src.vespa.datatypes import QueryResult
from src.vespa.retrieval.run import run_queries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> Dict[str, List[QueryResult]]:
    # Path to your config file
    config_path = (
        "/Users/vesaalexandru/Workspaces/cube/cube-publication/"
        "evaluate-ocr-rag-systems/src/vespa/vespa_config.yaml"
    )

    # Test queries
    queries = [
        "Percentage of non-fresh water as source?",
    ]

    try:
        logger.info("Starting query execution")
        results = await run_queries(
            config_path=config_path, queries=queries, display_results=True
        )
        logger.info("Query execution completed successfully")
        return results

    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    import base64
    from io import BytesIO

    import google.generativeai as genai
    from PIL import Image
    from vidore_benchmark.utils.image_utils import scale_image

    results = asyncio.run(main())

    genai.configure(api_key="AIzaSyCxMUFUaeApWRNr5HUS_xhWL26p0WLuG2w")

    queries = [
        "Percentage of non-fresh water as source?",
        # "Policies related to nature risk?",
        # "How much of produced water is recycled?",
    ]

    best_hit = results["Percentage of non-fresh water as source?"][0]
    pdf_url = best_hit.url
    pdf_title = best_hit.title
    # match_scores = best_hit["fields"]["matchfeatures"]["max_sim_per_page"]
    images = best_hit["fields"]["images"]
    sorted_pages = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
    best_page, score = sorted_pages[0]
    best_page = int(best_page)
    image_data = base64.b64decode(best_hit.source["fields"]["image"])
    image = Image.open(BytesIO(image_data))
    scaled_image = scale_image(image, 720)
    # # display(scaled_image)

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([queries[0], image])
    print(response)
