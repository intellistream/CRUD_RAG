import warnings
from typing import List, Optional

from haystack.nodes import RecentnessRanker
from haystack.schema import Document
from haystack.errors import NodeError


class RelevanceRecentnessRanker(RecentnessRanker):
    def __init__(
            self,
            date_meta_field: str,
            weight: float = 0.5,
            relevance_weight: float = 0.5,
            top_k: Optional[int] = None,
            ranking_mode: str = "reciprocal_rank_fusion",
    ):
        """
        Initializes the ranker to consider both relevance and recentness in document ranking.

        :param date_meta_field: Identifier for the date field in the document metadata.
        :param weight: Weight for recentness in the final score, ranging [0, 1].
        :param relevance_weight: Weight for relevance in the final score, ranging [0, 1].
        :param top_k: Optional; the number of top documents to return.
        :param ranking_mode: Mode for combining scores; supports "reciprocal_rank_fusion" or "score".
        """
        super().__init__(date_meta_field, weight, top_k, ranking_mode)
        self.relevance_weight = relevance_weight

        if self.relevance_weight < 0 or self.relevance_weight > 1:
            raise NodeError("Param <relevance_weight> must be in range [0, 1].")

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Extends predict to consider an additional relevance score from document metadata.
        """
        # First, call the parent predict method to rank documents based on recentness
        documents = super().predict(query, documents, top_k)

        # Adjust the score based on relevance, if relevance_weight > 0
        if self.relevance_weight > 0:
            for doc in documents:
                relevance_score = doc.meta.get("relevance_score", 0)
                if self.ranking_mode == "score":
                    # Assume existing doc.score is a combination of relevance and recency
                    # Adjust it according to the new relevance weight
                    doc.score = (doc.score * (1 - self.relevance_weight)) + (relevance_score * self.relevance_weight)
                elif self.ranking_mode == "reciprocal_rank_fusion":
                    # For RRF, this example simply demonstrates adjusting by relevance
                    # In practice, you'd need a more complex approach to integrate relevance into RRF
                    warnings.warn("RRF ranking mode might not fully integrate custom relevance scores.")

        # Sort documents by the adjusted score
        documents.sort(key=lambda doc: doc.score if doc.score is not None else -1, reverse=True)

        # Apply top_k limitation if specified
        if top_k is not None:
            documents = documents[:top_k]

        return documents
