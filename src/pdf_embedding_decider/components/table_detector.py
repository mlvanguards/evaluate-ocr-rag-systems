from types import Any, Dict, List

import fitz

from pdf_embedding_decider.interfaces import PDFAnalyzer
from src.pdf_embedding_decider.datatypes import AnalysisConfig, AnalysisResult


class TableDetector(PDFAnalyzer):
    """Analyzes the presence and structure of tables in the document"""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def _analyze_document(self, doc: fitz.Document) -> AnalysisResult:
        total_tables = 0
        table_details = []

        for page_num, page in enumerate(doc):
            tables = self._detect_tables(page)
            total_tables += len(tables)
            table_details.extend(
                [
                    {
                        "page": page_num + 1,
                        "rows": table["rows"],
                        "columns": table["columns"],
                        "area": table["area"],
                    }
                    for table in tables
                ]
            )

        return AnalysisResult(
            score=total_tables,
            details={"total_tables": total_tables, "table_details": table_details},
        )

    def _detect_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Detect tables based on text block alignment and spacing"""
        blocks = page.get_text("blocks")
        tables = []

        # Sort blocks by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # Sort by y, then x

        current_table = {"rows": [], "y_positions": set()}

        for block in sorted_blocks:
            x0, y0, x1, y1, text, *_ = block

            # Check if block is part of current table structure
            if current_table["rows"]:
                last_y = max(current_table["y_positions"])
                y_gap = y0 - last_y

                if y_gap > 20:  # New table or not part of table
                    if len(current_table["rows"]) >= self.config.table_row_threshold:
                        tables.append(self._finalize_table(current_table))
                    current_table = {"rows": [], "y_positions": set()}

            current_table["rows"].append({"text": text, "bbox": (x0, y0, x1, y1)})
            current_table["y_positions"].add(y0)

        # Check last table
        if len(current_table["rows"]) >= self.config.table_row_threshold:
            tables.append(self._finalize_table(current_table))

        return tables

    def _finalize_table(self, table_data: Dict) -> Dict[str, Any]:
        """Calculate table metrics"""
        rows = table_data["rows"]
        x_positions = []
        for row in rows:
            x_positions.extend([row["bbox"][0], row["bbox"][2]])

        # Estimate columns by analyzing x-position clusters
        x_clusters = self._cluster_positions(x_positions)

        return {
            "rows": len(rows),
            "columns": len(x_clusters)
            // 2,  # Divide by 2 as we counted start/end positions
            "area": self._calculate_table_area(rows),
        }

    @staticmethod
    def _cluster_positions(
        positions: List[float], threshold: float = 10
    ) -> List[float]:
        """Cluster similar x-positions together"""
        positions = sorted(positions)
        clusters = []
        current_cluster = [positions[0]]

        for pos in positions[1:]:
            if pos - current_cluster[-1] <= threshold:
                current_cluster.append(pos)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]

        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

    @staticmethod
    def _calculate_table_area(rows: List[Dict]) -> float:
        """Calculate the total area occupied by the table"""
        if not rows:
            return 0

        x_min = min(row["bbox"][0] for row in rows)
        x_max = max(row["bbox"][2] for row in rows)
        y_min = min(row["bbox"][1] for row in rows)
        y_max = max(row["bbox"][3] for row in rows)

        return (x_max - x_min) * (y_max - y_min)
