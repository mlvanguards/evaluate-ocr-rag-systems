import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from colpali_engine.models import ColQwen2, ColQwen2Processor
from torch.utils.data import DataLoader

from src.vespa.datatypes import QueryResult, VespaQueryConfig
from src.vespa.exceptions import VespaQueryError
from src.vespa.setup import VespaSetup
from vespa.deployment import VespaCloud
from vespa.io import VespaQueryResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class VespaQuerier:
    def __init__(
        self,
        config: VespaQueryConfig,
        setup: Optional[VespaSetup] = None,
        model_name: str = "vidore/colqwen2-v0.1",
    ):
        self.config = config
        self.setup = setup or VespaSetup(
            app_name=config.app_name, schema_config=config.schema_config
        )

        logger.info(f"Loading model {model_name}")
        self.model = ColQwen2.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.model.eval()

        self._init_vespa_cloud()

    def _init_vespa_cloud(self) -> None:
        try:
            self.vespa_cloud = VespaCloud(
                tenant=self.config.tenant_name,
                application=self.config.app_name,
                application_package=self.setup.app_package,
            )
        except Exception as e:
            logger.error(f"Failed to initialize VespaCloud: {str(e)}")
            raise VespaQueryError(f"VespaCloud initialization failed: {str(e)}")

    def _prepare_query_tensors(self, query: str) -> Dict[str, List[float]]:
        dataloader = DataLoader(
            [query],
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        with torch.no_grad():
            batch_query = next(iter(dataloader))
            batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
            embeddings_query = self.model(**batch_query)
            query_embedding = embeddings_query[0].cpu().float()

        # Convert embedding to list format
        embedding_list = query_embedding.tolist()
        if isinstance(embedding_list[0], list):
            # Handle case where embedding is 2D
            query_tensors = {i: emb for i, emb in enumerate(embedding_list)}
        else:
            # Handle case where embedding is 1D
            query_tensors = {0: embedding_list}

        return {"input.query(qt)": query_tensors}

    async def execute_queries(self, queries: List[str]) -> Dict[str, List[QueryResult]]:
        try:
            app = self.vespa_cloud.get_application()
            results: Dict[str, List[QueryResult]] = {}

            async with app.asyncio(
                connections=self.config.connections, timeout=self.config.timeout
            ) as session:
                for query in queries:
                    try:
                        logger.info(f"Executing query: {query}")
                        query_tensors = self._prepare_query_tensors(query)

                        logger.debug(f"Query tensors: {query_tensors}")

                        response = await session.query(
                            yql="select id, title, url, text, page_number, image from pdf_page where userInput(@userQuery)",
                            userQuery=query,
                            hits=self.config.hits_per_query,
                            body={
                                **query_tensors,
                                "presentation.timing": True,
                                "timeout": str(self.config.timeout),
                            },
                        )

                        if not response.is_successful():
                            error_msg = (
                                response.get_json()
                                if hasattr(response, "get_json")
                                else str(response)
                            )
                            logger.error(f"Query response error: {error_msg}")
                            raise VespaQueryError(f"Query failed: {error_msg}")

                        results[query] = self._process_response(response)

                    except Exception as e:
                        logger.error(f"Query failed for '{query}': {str(e)}")
                        results[query] = []

            return results

        except Exception as e:
            logger.error(f"Failed to execute queries: {str(e)}")
            raise VespaQueryError(f"Query execution failed: {str(e)}")

    def _process_response(self, response: VespaQueryResponse) -> List[QueryResult]:
        try:
            results = []
            for hit in response.hits:
                fields = hit["fields"]
                results.append(
                    QueryResult(
                        title=fields.get("title", "N/A"),
                        url=fields.get("url", "N/A"),
                        page_number=fields.get("page_number", -1),
                        relevance=float(hit.get("relevance", 0.0)),
                        text=fields.get("text", ""),
                        source=hit,
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return []

    def display_results(self, query: str, results: List[QueryResult]) -> None:
        print(f"\nQuery: {query}")
        print(f"Total Results: {len(results)}")

        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Page: {result.page_number}")
            print(f"Score: {result.relevance:.4f}")
            text_preview = (
                result.text[:200] + "..." if len(result.text) > 200 else result.text
            )
            print(f"Text Preview: {text_preview}")


async def run_queries(
    config_path: str, queries: List[str], display_results: bool = True
) -> Dict[str, List[QueryResult]]:
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if not config_data.get("vespa"):
            raise ValueError("Invalid configuration: 'vespa' section missing")

        vespa_config = VespaQueryConfig.from_dict(config_data["vespa"])
        querier = VespaQuerier(vespa_config)

        logger.info(f"Executing {len(queries)} queries")
        results = await querier.execute_queries(queries)

        if display_results:
            for query, query_results in results.items():
                querier.display_results(query, query_results)

        return results

    except Exception as e:
        logger.error(f"Failed to run queries: {str(e)}")
        raise


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
    import asyncio

    try:
        results = asyncio.run(main())

        # Print summary of results
        print("\nResults Summary:")
        for query, query_results in results.items():
            print(f"\nQuery: {query}")
            print(f"Number of results: {len(query_results)}")
            if query_results:
                print(f"Top result score: {query_results[0].relevance:.4f}")
                print(f"Top result title: {query_results[0].title}")

    except KeyboardInterrupt:
        logger.info("Query execution interrupted by user")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise
