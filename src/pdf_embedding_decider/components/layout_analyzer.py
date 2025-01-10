import logging
from types import Any, Dict

import fitz

from pdf_embedding_decider.interfaces import PDFAnalyzer
from src.pdf_embedding_decider.datatypes import AnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayoutAnalyzer(PDFAnalyzer):
    def _analyze_document(self, doc: fitz.Document) -> AnalysisResult:
        total_complexity = 0
        page_layouts = []

        for page_num, page in enumerate(doc):
            layout_info = self._analyze_page_layout(page)
            complexity = layout_info["block_count"] + len(layout_info["alignments"])
            total_complexity += complexity

            page_layouts.append(
                {"page": page_num + 1, **layout_info, "complexity_score": complexity}
            )

        return AnalysisResult(
            score=total_complexity,
            details={
                "total_complexity": total_complexity,
                "page_layouts": page_layouts,
            },
        )

    def _analyze_page_layout(self, page: fitz.Page) -> Dict[str, Any]:
        text_blocks = page.get_text("blocks")
        alignments = set()

        for block in text_blocks:
            x0, y0, x1, y1, *_ = block
            if x0 < page.rect.width * 0.3:
                alignments.add("left")
            elif x1 > page.rect.width * 0.7:
                alignments.add("right")
            else:
                alignments.add("center")

        return {"block_count": len(text_blocks), "alignments": list(alignments)}
