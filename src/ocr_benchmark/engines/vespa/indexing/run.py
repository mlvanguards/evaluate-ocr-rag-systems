import logging
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm
from vespa.application import Vespa
from vespa.deployment import VespaCloud

from src.ocr_benchmark.engines.vespa.indexing.pdf_processor import (
    PDFProcessingError,
    PDFProcessor,
)
from src.ocr_benchmark.engines.vespa.indexing.prepare_feed import (
    PDFPage,
    VespaFeedPreparator,
)
from src.ocr_benchmark.engines.vespa.setup import VespaSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VespaConfig:
    """Configuration for Vespa deployment."""

    app_name: str
    tenant_name: str
    connections: int = 1
    timeout: int = 180
    schema_name: str = "pdf_page"


@dataclass
class PDFInput:
    """Data class for PDF input information."""

    title: str
    url: str


class VespaDeploymentError(Exception):
    """Custom exception for Vespa deployment errors."""

    pass


class VespaDeployer:
    """Class for handling Vespa Cloud deployment and data feeding."""

    def __init__(
        self,
        config: VespaConfig,
        pdf_processor: Optional[PDFProcessor] = None,
        feed_preparator: Optional[VespaFeedPreparator] = None,
    ) -> None:
        """
        Initialize VespaDeployer with configuration and processors.

        Args:
            config: VespaConfig object containing deployment settings
            pdf_processor: Optional PDFProcessor instance
            feed_preparator: Optional VespaFeedPreparator instance
        """
        self.config = config
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.feed_preparator = feed_preparator or VespaFeedPreparator()

    async def deploy_and_feed(self, vespa_feed: List[PDFPage]) -> Vespa:
        """
        Deploy application to Vespa Cloud and feed data.

        Args:
            vespa_feed: List of PDFPage objects to be indexed

        Returns:
            Deployed Vespa application instance

        Raises:
            VespaDeploymentError: If deployment or feeding fails
        """
        try:
            logger.info("Setting up Vespa deployment")
            vespa_setup = VespaSetup(self.config.app_name)

            vespa_cloud = VespaCloud(
                tenant=self.config.tenant_name,
                application=self.config.app_name,
                application_package=vespa_setup.app_package,
            )

            logger.info("Deploying to Vespa Cloud")
            app = vespa_cloud.deploy()

            await self._feed_data(app, vespa_feed)

            return app

        except Exception as e:
            logger.error(f"Vespa deployment failed: {str(e)}")
            raise VespaDeploymentError(f"Deployment failed: {str(e)}")

    async def _feed_data(self, app: Vespa, vespa_feed: List[PDFPage]) -> None:
        """
        Feed data to Vespa application.

        Args:
            app: Deployed Vespa application instance
            vespa_feed: List of PDFPage objects to be indexed

        Raises:
            VespaDeploymentError: If feeding fails
        """
        failed_documents: List[Dict[str, Any]] = []

        async with app.asyncio(
            connections=self.config.connections, timeout=self.config.timeout
        ) as session:
            logger.info("Starting data feed process")

            for page in tqdm(vespa_feed, desc="Feeding documents"):
                try:
                    response = await session.feed_data_point(
                        data_id=page.id,
                        fields=page.__dict__,
                        schema=self.config.schema_name,
                    )

                    if not response.is_successful():
                        failed_documents.append(
                            {"id": page.id, "error": response.json()}
                        )
                        logger.warning(
                            f"Failed to feed document {page.id}: {response.json()}"
                        )

                except Exception as e:
                    failed_documents.append({"id": page.id, "error": str(e)})
                    logger.error(f"Error feeding document {page.id}: {str(e)}")

            if failed_documents:
                logger.error(f"Failed to feed {len(failed_documents)} documents")
                self._save_failed_documents(failed_documents)
                raise VespaDeploymentError("Some documents failed to feed")

    @staticmethod
    def _save_failed_documents(failed_docs: List[Dict[str, Any]]) -> None:
        """
        Save failed document information to a file.

        Args:
            failed_docs: List of failed document information
        """
        output_dir = Path("logs/failed_documents")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"failed_documents_{int(time.time())}.yaml"

        with open(output_file, "w") as f:
            yaml.dump(failed_docs, f)

        logger.info(f"Failed documents information saved to {output_file}")


async def run_indexing(
    config_path: str,
    pdfs: List[PDFInput],
    pdf_processor: Optional[PDFProcessor] = None,
    feed_preparator: Optional[VespaFeedPreparator] = None,
) -> None:
    """
    Run the indexing process for a given list of PDFs.

    Args:
        config_path: Path to Vespa configuration file
        pdfs: List of PDFInput objects containing PDF information
        pdf_processor: Optional custom PDFProcessor instance
        feed_preparator: Optional custom VespaFeedPreparator instance

    Raises:
        FileNotFoundError: If configuration file is not found
        PDFProcessingError: If PDF processing fails
        VespaDeploymentError: If Vespa deployment or feeding fails
        ValueError: If input parameters are invalid
    """
    try:
        # Validate inputs
        if not pdfs:
            raise ValueError("PDF list cannot be empty")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if not config_data.get("vespa"):
            raise ValueError("Invalid configuration: 'vespa' section missing")

        vespa_config = VespaConfig(**config_data["vespa"])

        # Initialize processors if not provided
        pdf_processor = pdf_processor or PDFProcessor()
        feed_preparator = feed_preparator or VespaFeedPreparator()
        deployer = VespaDeployer(vespa_config, pdf_processor, feed_preparator)

        # Convert PDFInput objects to dictionary format
        pdf_data = [{"title": pdf.title, "url": pdf.url} for pdf in pdfs]

        # Process PDFs
        logger.info(f"Processing {len(pdfs)} PDFs")
        processed_data = pdf_processor.process_pdf(pdf_data)

        # Prepare Vespa feed
        logger.info("Preparing Vespa feed")
        vespa_feed = feed_preparator.prepare_feed(processed_data)

        # Deploy and feed data
        logger.info("Starting deployment and feeding process")
        await deployer.deploy_and_feed(vespa_feed)

        logger.info("Indexing process completed successfully")

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise
    except (PDFProcessingError, VespaDeploymentError) as e:
        logger.error(f"Processing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {str(e)}")
        raise


# Example usage:
async def index_documents():
    """Example function showing how to use run_indexing."""
    config_path = "vespa_config.yaml"

    pdfs = [
        PDFInput(
            title="Building a Resilient Strategy",
            url="https://example.com/document1.pdf",
        ),
        PDFInput(
            title="Energy Transition Report", url="https://example.com/document2.pdf"
        ),
    ]

    try:
        await run_indexing(config_path, pdfs)
    except Exception as e:
        logger.error(f"Failed to index documents: {str(e)}")
        raise
