import logging
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from src.vespa.datatypes import (
    PDFInput,
    VespaConfig,
    VespaSchemaConfig,
)
from src.vespa.indexing.pdf_processor import PDFProcessor
from src.vespa.indexing.prepare_feed import VespaFeedPreparator
from src.vespa.setup import VespaSetup
from vespa.application import Vespa
from vespa.deployment import VespaCloud

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class VespaDeployer:
    def __init__(
        self,
        config: VespaConfig,
        pdf_processor: Optional[PDFProcessor] = None,
        feed_preparator: Optional[VespaFeedPreparator] = None,
    ) -> None:
        self.config = config
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.feed_preparator = feed_preparator or VespaFeedPreparator()

    async def _feed_data(self, app: Vespa, vespa_feed: List[Dict]) -> None:
        failed_documents = []

        async with app.asyncio(
            connections=self.config.connections, timeout=self.config.timeout
        ) as session:
            logger.info("Starting data feed process")

            for doc in tqdm(vespa_feed, desc="Feeding documents"):
                try:
                    logger.debug(f"Feeding document: {doc['fields']['url']}")

                    response = await session.feed_data_point(
                        schema=self.config.schema_name,
                        data_id=doc["fields"]["url"],
                        fields=doc["fields"],
                    )

                    if not response.is_successful():
                        error_msg = (
                            response.get_json()
                            if response.get_json()
                            else str(response)
                        )
                        logger.error(f"Feed failed: {error_msg}")
                        failed_documents.append(
                            {"id": doc["fields"]["url"], "error": error_msg}
                        )

                except Exception as e:
                    logger.error(f"Feed error for {doc['fields']['url']}: {str(e)}")
                    failed_documents.append(
                        {"id": doc["fields"]["url"], "error": str(e)}
                    )

            if failed_documents:
                self._save_failed_documents(failed_documents)
                error_details = "\n".join(
                    [f"Doc {doc['id']}: {doc['error']}" for doc in failed_documents]
                )
                raise Exception(f"Documents failed to feed:\n{error_details}")

    @staticmethod
    def _save_failed_documents(failed_docs: List[Dict[str, Any]]) -> None:
        output_dir = Path("logs/failed_documents")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"failed_documents_{int(time())}.yaml"
        with open(output_file, "w") as f:
            yaml.dump(failed_docs, f)
        logger.info(f"Saved failed documents to {output_file}")

    async def deploy_and_feed(self, vespa_feed: List[Dict]) -> Vespa:
        try:
            vespa_setup = VespaSetup(
                app_name=self.config.app_name,
                schema_config=self.config.schema_config or VespaSchemaConfig(),
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
            raise


async def run_indexing(
    config_path: str,
    pdfs: List[PDFInput],
    pdf_processor: Optional[PDFProcessor] = None,
    feed_preparator: Optional[VespaFeedPreparator] = None,
) -> None:
    try:
        if not pdfs:
            raise ValueError("PDF list cannot be empty")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if not config_data.get("vespa"):
            raise ValueError("Invalid configuration: 'vespa' section missing")

        schema_config = config_data["vespa"].get("schema")
        vespa_config = VespaConfig(
            app_name=config_data["vespa"]["app_name"],
            tenant_name=config_data["vespa"]["tenant_name"],
            connections=config_data["vespa"].get("connections", 1),
            timeout=config_data["vespa"].get("timeout", 180),
            schema_name=config_data["vespa"].get("schema_name", "pdf_page"),
            schema_config=VespaSchemaConfig.from_dict(schema_config)
            if schema_config
            else None,
        )

        deployer = VespaDeployer(
            config=vespa_config,
            pdf_processor=pdf_processor,
            feed_preparator=feed_preparator,
        )

        processed_data = pdf_processor.process_pdf(
            [{"title": pdf.title, "url": pdf.url} for pdf in pdfs]
        )
        vespa_feed = feed_preparator.prepare_feed(processed_data)

        await deployer.deploy_and_feed(vespa_feed)
        logger.info("Indexing process completed successfully")

    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise
