import logging
from typing import List, Optional

from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    FirstPhaseRanking,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)

from src.ocr_benchmark.engines.vespa.datatypes import VespaSchemaConfig
from src.ocr_benchmark.engines.vespa.exceptions import VespaSetupError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VespaSetup:
    """Class for setting up Vespa application package and schema."""

    def __init__(
        self, app_name: str, schema_config: Optional[VespaSchemaConfig] = None
    ) -> None:
        """
        Initialize Vespa setup.

        Args:
            app_name: Name of the Vespa application
            schema_config: Optional configuration for schema settings
        """
        self.app_name = app_name
        self.config = schema_config or VespaSchemaConfig()

        try:
            logger.info(f"Creating Vespa schema for application: {app_name}")
            self.schema = self._create_schema()
            self.app_package = ApplicationPackage(name=app_name, schema=[self.schema])
        except Exception as e:
            logger.error(f"Failed to create Vespa setup: {str(e)}")
            raise VespaSetupError(f"Setup failed: {str(e)}")

    def _create_schema(self) -> Schema:
        """
        Create the Vespa schema with fields and rank profiles.

        Returns:
            Configured Schema object

        Raises:
            VespaSetupError: If schema creation fails
        """
        try:
            schema = Schema(
                name="pdf_page",
                document=Document(fields=self._create_fields()),
                fieldsets=[FieldSet(name="default", fields=["title", "text"])],
            )

            self._add_rank_profiles(schema)
            return schema

        except Exception as e:
            logger.error(f"Failed to create schema: {str(e)}")
            raise VespaSetupError(f"Schema creation failed: {str(e)}")

    def _create_fields(self) -> List[Field]:
        """
        Create the fields for the schema.

        Returns:
            List of Field objects
        """
        return [
            Field(
                name="id", type="string", indexing=["summary", "index"], match=["word"]
            ),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(
                name="title",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(name="page_number", type="int", indexing=["summary", "attribute"]),
            Field(name="image", type="raw", indexing=["summary"]),
            Field(
                name="text",
                type="string",
                indexing=["index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="embedding",
                type=f"tensor<int8>(patch{{}}, v[{self.config.tensor_dimensions}])",
                indexing=["attribute", "index"],
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=self.config.hnsw_max_links,
                    neighbors_to_explore_at_insert=self.config.hnsw_neighbors,
                ),
            ),
        ]

    def _add_rank_profiles(self, schema: Schema) -> None:
        """
        Add rank profiles to the schema.

        Args:
            schema: Schema object to add rank profiles to
        """
        # Add default ranking profile
        schema.add_rank_profile(self._create_default_profile())

        # Add retrieval and rerank profile
        schema.add_rank_profile(self._create_retrieval_rerank_profile())

    def _create_default_profile(self) -> RankProfile:
        """
        Create the default rank profile.

        Returns:
            Configured RankProfile object
        """
        return RankProfile(
            name="default",
            inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
            functions=[
                Function(
                    name="max_sim",
                    expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                    """,
                ),
                Function(name="bm25_score", expression="bm25(title) + bm25(text)"),
            ],
            first_phase=FirstPhaseRanking(expression="bm25_score"),
            second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=100),
        )

    def _create_retrieval_rerank_profile(self) -> RankProfile:
        """
        Create the retrieval and rerank profile.

        Returns:
            Configured RankProfile object
        """
        input_query_tensors = []

        # Add query tensors for each term
        for i in range(self.config.max_query_terms):
            input_query_tensors.append(
                (f"query(rq{i})", f"tensor<int8>(v[{self.config.tensor_dimensions}])")
            )

        # Add additional query tensors
        input_query_tensors.extend(
            [
                ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                (
                    "query(qtb)",
                    f"tensor<int8>(querytoken{{}}, v[{self.config.tensor_dimensions}])",
                ),
            ]
        )

        return RankProfile(
            name="retrieval-and-rerank",
            inputs=input_query_tensors,
            functions=[
                Function(
                    name="max_sim",
                    expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                    """,
                ),
                Function(
                    name="max_sim_binary",
                    expression="""
                    sum(
                        reduce(
                            1/(1 + sum(
                                hamming(query(qtb), attribute(embedding)) ,v)
                            ),
                            max,
                            patch
                        ),
                        querytoken
                    )
                    """,
                ),
            ],
            first_phase=FirstPhaseRanking(expression="max_sim_binary"),
            second_phase=SecondPhaseRanking(
                expression="max_sim", rerank_count=self.config.rerank_count
            ),
        )
