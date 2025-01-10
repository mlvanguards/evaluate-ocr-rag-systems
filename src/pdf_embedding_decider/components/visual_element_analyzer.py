import fitz

from pdf_embedding_decider.interfaces import PDFAnalyzer
from src.pdf_embedding_decider.datatypes import AnalysisConfig, AnalysisResult


class VisualElementAnalyzer(PDFAnalyzer):
    def __init__(self, config: AnalysisConfig):
        self.config = config

    def _analyze_document(self, doc: fitz.Document) -> AnalysisResult:
        total_visual_elements = 0
        image_details = []
        drawing_details = []

        for page_num, page in enumerate(doc):
            # Analyze images
            images = page.get_images(full=True)
            filtered_images = [
                img for img in images if self._is_significant_image(page, img)
            ]
            image_details.extend(
                [
                    {"page": page_num + 1, "size": self._get_image_size(page, img)}
                    for img in filtered_images
                ]
            )

            # Analyze vector graphics
            drawings = page.get_drawings()
            significant_drawings = self._filter_significant_drawings(drawings)
            drawing_details.extend(
                [
                    {"page": page_num + 1, "complexity": len(draw["items"])}
                    for draw in significant_drawings
                ]
            )

            total_visual_elements += len(filtered_images) + len(significant_drawings)

        return AnalysisResult(
            score=total_visual_elements,
            details={
                "total_elements": total_visual_elements,
                "images": image_details,
                "drawings": drawing_details,
            },
        )

    def _is_significant_image(self, page: fitz.Page, image: tuple) -> bool:
        """Filter out small or insignificant images"""
        xref = image[0]
        pix = fitz.Pixmap(page.parent, xref)
        area = pix.width * pix.height
        return area >= self.config.min_image_size

    def _filter_significant_drawings(self, drawings: List[dict]) -> List[dict]:
        """Filter out simple decorative elements"""
        return [d for d in drawings if len(d["items"]) > 2]

    @staticmethod
    def _get_image_size(page: fitz.Page, image: tuple) -> dict:
        xref = image[0]
        pix = fitz.Pixmap(page.parent, xref)
        return {"width": pix.width, "height": pix.height}
