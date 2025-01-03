import asyncio
import logging
from typing import Dict, List

from src.ocr_benchmark.engines.vespa.datatypes import QueryResult
from src.ocr_benchmark.engines.vespa.retrieval.run import run_queries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> Dict[str, List[QueryResult]]:
    config_path = (
        "/Users/vesaalexandru/Workspaces/cube/cube-publication/"
        "evaluate-ocr-rag-systems/src/ocr_benchmark/engines/vespa/vespa_config.yaml"
    )

    queries = [
        "Percentage of non-fresh water as source?",
        # "Policies related to nature risk?",
        # "How much of produced water is recycled?",
    ]

    try:
        logger.info("Starting query execution")
        results = await run_queries(config_path, queries, display_results=True)
        logger.info("Query execution completed")
        return results

    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    results = asyncio.run(main())
