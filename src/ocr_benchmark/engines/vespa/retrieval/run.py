import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from colpali_engine.models import ColQwen2, ColQwen2Processor
from torch.utils.data import DataLoader
from vespa.deployment import VespaCloud
from vespa.io import VespaQueryResponse

from src.ocr_benchmark.engines.vespa.datatypes import QueryResult, VespaQueryConfig
from src.ocr_benchmark.engines.vespa.exceptions import VespaQueryError
from src.ocr_benchmark.engines.vespa.setup import VespaSetup

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

    def _prepare_query_tensors(self, query: str) -> Dict[str, Any]:
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

        query_tensors = {
            "input.query(qt)": {k: v.tolist() for k, v in enumerate(query_embedding)},
        }

        # Add binary query tensors for nearest neighbor search
        binary_query = np.packbits(query_embedding.numpy() > 0, axis=1).astype(np.int8)
        for i in range(len(binary_query)):
            query_tensors[f"input.query(rq{i})"] = binary_query[i].tolist()

        return query_tensors

    def _build_query_string(self, query: str) -> str:
        nn_parts = []
        target_hits = self.config.hits_per_query * 2
        for i in range(self.config.tensor_dimensions):
            nn_parts.append(
                f"({{targetHits:{target_hits}}}nearestNeighbor(embedding,rq{i}))"
            )
        return " OR ".join(nn_parts)

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

                        response = await session.query(
                            yql=(
                                "select id, title, url, text, page_number, image "
                                f"from pdf_page where {self._build_query_string(query)}"
                            ),
                            ranking="default",
                            hits=self.config.hits_per_query,
                            body={
                                **query_tensors,
                                "ranking.profile": "two-phase",
                                "ranking.features.query(bm25)": True,
                                "presentation.timing": True,
                            },
                        )

                        if not response.is_successful():
                            raise VespaQueryError(f"Query failed: {response.json()}")

                        results[query] = self._process_response(response)

                    except Exception as e:
                        logger.error(f"Query failed for '{query}': {str(e)}")
                        results[query] = []

            return results

        except Exception as e:
            logger.error(f"Failed to execute queries: {str(e)}")
            raise VespaQueryError(f"Query execution failed: {str(e)}")

    def _process_response(self, response: VespaQueryResponse) -> List[QueryResult]:
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
