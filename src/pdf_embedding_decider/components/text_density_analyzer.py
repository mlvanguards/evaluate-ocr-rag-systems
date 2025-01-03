import fitz

from src.pdf_embedding_decider.analyzer import PDFAnalyzer
from src.pdf_embedding_decider.datatypes import AnalysisConfig, AnalysisResult


class TextDensityAnalyzer(PDFAnalyzer):
    def __init__(self, config: AnalysisConfig):
        self.config = config

    def _analyze_document(self, doc: fitz.Document) -> AnalysisResult:
        total_density = 0
        page_densities = []

        for page_num, page in enumerate(doc):
            text_area = self._calculate_text_area(page)
            page_area = page.rect.width * page.rect.height
            density = text_area / page_area if page_area > 0 else 0

            page_densities.append(
                {
                    "page": page_num + 1,
                    "density": density,
                    "text_area": text_area,
                    "page_area": page_area,
                }
            )
            total_density += density

        avg_density = total_density / len(doc) if len(doc) > 0 else 0
        return AnalysisResult(
            score=avg_density,
            details={"average_density": avg_density, "page_densities": page_densities},
        )

    @staticmethod
    def _calculate_text_area(page: fitz.Page) -> float:
        text_area = 0
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, *_ = block
            text_area += (x1 - x0) * (y1 - y0)
        return text_area
