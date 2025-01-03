import logging

import fitz

from src.pdf_embedding_decider.datatypes import AnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFAnalyzer:
    """Base class for PDF analysis with error handling and resource management"""

    def analyze(self, pdf_path: str) -> AnalysisResult:
        try:
            doc = fitz.open(pdf_path)
            result = self._analyze_document(doc)
            return result
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            raise
        finally:
            if "doc" in locals():
                doc.close()

    def _analyze_document(self, doc: fitz.Document) -> AnalysisResult:
        raise NotImplementedError
