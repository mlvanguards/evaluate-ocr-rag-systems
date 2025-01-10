from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch import Tensor


@dataclass
class PDFInput:
    """Data class for PDF input information."""

    title: str
    url: str


@dataclass
class PDFData:
    """Data class to store processed PDF information."""

    url: str
    title: str
    images: List[Any]  # PIL.Image type
    texts: List[str]
    embeddings: List[Tensor]


@dataclass
class PDFPage:
    id: str
    url: str
    title: str
    page_number: int
    image: str
    text: str
    embedding: Dict[int, str]


@dataclass
class VespaSchemaConfig:
    """Configuration for Vespa schema settings."""

    max_query_terms: int = 64
    hnsw_max_links: int = 32
    hnsw_neighbors: int = 400
    rerank_count: int = 10
    tensor_dimensions: int = 16

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VespaSchemaConfig":
        """Create schema config from dictionary with default values."""
        return cls(
            max_query_terms=config_dict.get("max_query_terms", 64),
            hnsw_max_links=config_dict.get("hnsw_max_links", 32),
            hnsw_neighbors=config_dict.get("hnsw_neighbors", 400),
            rerank_count=config_dict.get("rerank_count", 10),
            tensor_dimensions=config_dict.get("tensor_dimensions", 16),
        )


@dataclass
class VespaConfig:
    """Configuration for Vespa deployment."""

    app_name: str
    tenant_name: str
    connections: int = 1
    timeout: int = 180
    schema_name: str = "pdf_page"
    schema_config: Optional[VespaSchemaConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VespaConfig":
        """Create VespaConfig from dictionary, handling schema config separately."""
        # Extract and convert schema config if present
        schema_dict = config_dict.pop("schema", {})
        schema_config = (
            VespaSchemaConfig.from_dict(schema_dict) if schema_dict else None
        )

        return cls(**config_dict, schema_config=schema_config)


@dataclass
class VespaQueryConfig:
    app_name: str
    tenant_name: str
    connections: int = 1
    timeout: int = 180
    hits_per_query: int = 5
    schema_name: str = "pdf_page"
    tensor_dimensions: int = 16  # Added this field
    schema_config: Optional[VespaSchemaConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VespaQueryConfig":
        schema_dict = config_dict.pop("schema", {})
        schema_config = (
            VespaSchemaConfig.from_dict(schema_dict) if schema_dict else None
        )
        return cls(
            **{k: v for k, v in config_dict.items() if k != "schema"},
            schema_config=schema_config,
        )


@dataclass
class QueryResult:
    """Data class for query results."""

    title: str
    url: str
    page_number: int
    relevance: float
    text: str
    source: Dict[str, Any]
