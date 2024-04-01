from haystack.nodes import SentenceTransformersRanker, BaseRanker

from src.components.reranker.simple_reranker import RelevanceRecentnessRanker


class RerankerFactory:
    reranker_classes = {
        "simple": RelevanceRecentnessRanker,
        "transformer": SentenceTransformersRanker
        # Add other rerankers here
    }

    @classmethod
    def get_reranker(cls, reranker_type: str, **kwargs) -> BaseRanker:
        if reranker_type not in cls.reranker_classes:
            raise ValueError(f"Reranker type '{reranker_type}' is not supported.")

        return cls.reranker_classes[reranker_type](**kwargs)
