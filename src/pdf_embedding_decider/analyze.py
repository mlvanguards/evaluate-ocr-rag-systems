import logging
from typing import Dict

from src.pdf_embedding_decider.components.layout_analyzer import LayoutAnalyzer
from src.pdf_embedding_decider.components.table_detector import TableDetector
from src.pdf_embedding_decider.components.text_density_analyzer import (
    TextDensityAnalyzer,
)
from src.pdf_embedding_decider.components.visual_element_analyzer import (
    VisualElementAnalyzer,
)
from src.pdf_embedding_decider.datatypes import (
    AnalysisConfig,
    AnalysisResult,
    EmbeddingType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFEmbeddingDecider:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzers = {
            "visual": VisualElementAnalyzer(config),
            "density": TextDensityAnalyzer(config),
            "layout": LayoutAnalyzer(),
            "table": TableDetector(config),
        }

    def analyze(self, pdf_path: str) -> Dict[str, AnalysisResult]:
        """Run all analyses and return detailed results"""
        return {
            name: analyzer.analyze(pdf_path)
            for name, analyzer in self.analyzers.items()
        }

    def decide(self, pdf_path: str) -> EmbeddingType:
        """Determine the appropriate embedding type based on PDF analysis"""
        try:
            results = self.analyze(pdf_path)

            # Log detailed analysis results
            logger.info("Analysis Results:")
            for analyzer_name, result in results.items():
                logger.info(f"{analyzer_name}: {result}")

            # Enhanced decision logic incorporating table analysis
            table_score = results["table"].score

            if table_score > 0:
                table_influence = min(1.0, table_score * self.config.table_weight)
                adjusted_density_threshold = self.config.text_density_threshold * (
                    1 - table_influence
                )
            else:
                adjusted_density_threshold = self.config.text_density_threshold

            if (
                results["visual"].score > self.config.visual_threshold
                or results["density"].score < adjusted_density_threshold
                or (
                    results["layout"].score > self.config.layout_threshold
                    and table_score == 0
                )
            ):  # Only consider complex layout if not tabular
                return EmbeddingType.COLPALI

            return EmbeddingType.TRADITIONAL

        except Exception as e:
            logger.error(f"Error deciding embedding type: {str(e)}")
            raise


if __name__ == "__main__":
    config = AnalysisConfig(
        visual_threshold=15,
        text_density_threshold=0.25,
        layout_threshold=100,
        min_image_size=1000,
        table_row_threshold=5,
        table_weight=0.3,
    )

    # Initialize decider
    decider = PDFEmbeddingDecider(config)

    # Example usage
    pdf_path = "/Users/vesaalexandru/Workspaces/cube/cube-publication/evaluate-ocr-rag-systems/data/aiminded-extras-octomrbie-decembrie-2023.pdf"
    # pdf_path = "/Users/vesaalexandru/Workspaces/cube/cube-publication/evaluate-ocr-rag-systems/data/paper01-1-2.pdf"
    try:
        # Get detailed analysis
        analysis_results = decider.analyze(pdf_path)

        # Get final decision
        embedding_type = decider.decide(pdf_path)

        logger.info(f"Recommended embedding type: {embedding_type.value}")
        logger.info("Detailed analysis results:")
        for analyzer_name, result in analysis_results.items():
            logger.info(f"{analyzer_name}: {result}")

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
