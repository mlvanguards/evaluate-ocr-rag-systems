import logging
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm
from vespa.application import Vespa
from vespa.deployment import VespaCloud

from src.ocr_benchmark.engines.vespa.datatypes import PDFInput, VespaConfig
from src.ocr_benchmark.engines.vespa.exceptions import VespaDeploymentError
from src.ocr_benchmark.engines.vespa.indexing.pdf_processor import (
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


class VespaDeployer:
    def __init__(
        self,
        config: VespaConfig,
        pdf_processor: Optional[PDFProcessor] = None,
        feed_preparator: Optional[VespaFeedPreparator] = None,
    ) -> None:
        """Initialize VespaDeployer with configuration and processors."""
        self.config = config
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.feed_preparator = feed_preparator or VespaFeedPreparator()

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

        output_file = output_dir / f"failed_documents_{int(time())}.yaml"

        with open(output_file, "w") as f:
            yaml.dump(failed_docs, f)

        logger.info(f"Failed documents information saved to {output_file}")

    async def deploy_and_feed(self, vespa_feed: List[PDFPage]) -> Vespa:
        """Deploy application to Vespa Cloud and feed data."""
        try:
            logger.info("Setting up Vespa deployment")
            vespa_setup = VespaSetup(
                app_name=self.config.app_name, schema_config=self.config.schema_config
            )

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


async def run_indexing(
    config_path: str,
    pdfs: List[PDFInput],
    pdf_processor: Optional[PDFProcessor] = None,
    feed_preparator: Optional[VespaFeedPreparator] = None,
) -> None:
    """
    Run the indexing process for a given list of PDFs.
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

        # Use the new from_dict method to create config
        vespa_config = VespaConfig.from_dict(config_data["vespa"])

        # Initialize processors
        if pdf_processor is None:
            pdf_processor = PDFProcessor()

        if feed_preparator is None:
            feed_preparator = VespaFeedPreparator()

        # Create deployer
        deployer = VespaDeployer(
            config=vespa_config,
            pdf_processor=pdf_processor,
            feed_preparator=feed_preparator,
        )

        # Process PDFs
        pdf_data = [{"title": pdf.title, "url": pdf.url} for pdf in pdfs]

        logger.info(f"Processing {len(pdfs)} PDFs")
        processed_data = pdf_processor.process_pdf(pdf_data)

        if not processed_data:
            raise ValueError("No data processed from PDFs")

        logger.info("Preparing Vespa feed")
        vespa_feed = feed_preparator.prepare_feed(processed_data)

        if not vespa_feed:
            raise ValueError("No feed data prepared")

        logger.info("Starting deployment and feeding process")
        await deployer.deploy_and_feed(vespa_feed)

        logger.info("Indexing process completed successfully")

    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise
