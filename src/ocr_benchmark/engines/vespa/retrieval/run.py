import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from vespa.deployment import VespaCloud
from vespa.io import VespaQueryResponse

from src.ocr_benchmark.engines.vespa.setup import VespaSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VespaQueryConfig:
    """Configuration for Vespa queries."""

    tenant_name: str
    app_name: str
    schema_name: str = "pdf_page"  # Add this field with default value
    connections: int = 1
    timeout: int = 180
    query_timeout: int = 120
    hits_per_query: int = 5


@dataclass
class QueryResult:
    """Data class to store query result information."""

    title: str
    url: str
    page_number: int
    relevance: float
    source: Dict[str, Any]


class VespaQueryError(Exception):
    """Custom exception for Vespa query errors."""

    pass


class VespaQuerier:
    """Class for handling Vespa queries and result processing."""

    def __init__(
        self, config: VespaQueryConfig, setup: Optional[VespaSetup] = None
    ) -> None:
        """
        Initialize the Vespa querier.

        Args:
            config: VespaQueryConfig object containing query settings
            setup: Optional VespaSetup instance
        """
        self.config = config
        self.setup = setup or VespaSetup(config.app_name)
        self._init_vespa_cloud()

    def _init_vespa_cloud(self) -> None:
        """Initialize VespaCloud connection."""
        try:
            self.vespa_cloud = VespaCloud(
                tenant=self.config.tenant_name,
                application=self.config.app_name,
                application_package=self.setup.app_package,
            )
        except Exception as e:
            logger.error(f"Failed to initialize VespaCloud: {str(e)}")
            raise VespaQueryError(f"VespaCloud initialization failed: {str(e)}")

    async def execute_queries(self, queries: List[str]) -> Dict[str, List[QueryResult]]:
        """
        Execute multiple queries and return results.

        Args:
            queries: List of query strings to execute

        Returns:
            Dictionary mapping queries to their results

        Raises:
            VespaQueryError: If query execution fails
        """
        try:
            app = self.vespa_cloud.get_application()
            results: Dict[str, List[QueryResult]] = {}

            async with app.asyncio(
                connections=self.config.connections, timeout=self.config.timeout
            ) as session:
                for query in queries:
                    try:
                        query_results = await self._execute_single_query(session, query)
                        results[query] = query_results
                    except Exception as e:
                        logger.error(f"Query failed for '{query}': {str(e)}")
                        results[query] = []

            return results

        except Exception as e:
            logger.error(f"Failed to execute queries: {str(e)}")
            raise VespaQueryError(f"Query execution failed: {str(e)}")

    async def _execute_single_query(
        self, session: Any, query: str
    ) -> List[QueryResult]:
        """
        Execute a single query and process its results.

        Args:
            session: Vespa session object
            query: Query string to execute

        Returns:
            List of QueryResult objects

        Raises:
            VespaQueryError: If query fails
        """
        try:
            response: VespaQueryResponse = await session.query(
                yql="select title, url, image, page_number from pdf_page where userInput(@userQuery)",
                ranking="default",
                userQuery=query,
                timeout=self.config.query_timeout,
                hits=self.config.hits_per_query,
                body={"presentation.timing": True},
            )

            if not response.is_successful():
                raise VespaQueryError(f"Query failed: {response.json()}")

            return self._process_query_results(response)

        except Exception as e:
            logger.error(f"Failed to execute query '{query}': {str(e)}")
            raise VespaQueryError(f"Query execution failed: {str(e)}")

    def _process_query_results(self, response: VespaQueryResponse) -> List[QueryResult]:
        """
        Process query response into QueryResult objects.

        Args:
            response: VespaQueryResponse object

        Returns:
            List of QueryResult objects
        """
        results = []
        for hit in response.hits:
            fields = hit["fields"]
            results.append(
                QueryResult(
                    title=fields.get("title", "N/A"),
                    url=fields.get("url", "N/A"),
                    page_number=fields.get("page_number", -1),
                    relevance=hit.get("relevance", 0.0),
                    source=hit,
                )
            )
        return results

    def display_results(self, query: str, results: List[QueryResult]) -> None:
        """
        Display query results in a formatted manner.

        Args:
            query: Original query string
            results: List of QueryResult objects
        """
        print(f"\nQuery: {query}")
        print(f"Total Results: {len(results)}")

        for idx, result in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Page: {result.page_number}")
            print(f"Relevance Score: {result.relevance:.4f}")


async def run_queries(
    config_path: str, queries: List[str], display_results: bool = True
) -> Dict[str, List[QueryResult]]:
    """
    Run queries using configuration from file.

    Args:
        config_path: Path to configuration file
        queries: List of queries to execute
        display_results: Whether to print results

    Returns:
        Dictionary of query results

    Raises:
        FileNotFoundError: If configuration file is not found
        VespaQueryError: If query execution fails
    """
    try:
        # Load configuration
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Initialize querier
        config = VespaQueryConfig(**config_data["vespa"])
        querier = VespaQuerier(config)

        # Execute queries
        results = await querier.execute_queries(queries)

        # Display results if requested
        if display_results:
            for query, query_results in results.items():
                querier.display_results(query, query_results)

        return results

    except Exception as e:
        logger.error(f"Failed to run queries: {str(e)}")
        raise


async def main():
    """Main entry point for the application."""
    config_path = "vespa_config.yaml"

    queries = [
        "Percentage of non-fresh water as source?",
        "Policies related to nature risk?",
        "How much of produced water is recycled?",
    ]

    try:
        await run_queries(config_path, queries)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
