from typing import Type, Dict, Optional
from haystack.document_stores import BaseDocumentStore  # Use BaseDocumentStore for type hinting
from haystack.nodes import BaseRetriever, DensePassageRetriever, BM25Retriever, EmbeddingRetriever

from src.components.retrievers.rania_retriever import RANIARetriever


class RetrieverFactory:
    retriever_classes: Dict[str, Type[BaseRetriever]] = {
        "bm25": BM25Retriever,
        "dpr": DensePassageRetriever,
        "embedding": EmbeddingRetriever,
        "rania": RANIARetriever,
    }

    @classmethod
    def get_retriever(cls, retriever_type: str, document_store: BaseDocumentStore,
                      query_embedding_model: Optional[str] = None, passage_embedding_model: Optional[str] = None,
                      **kwargs) -> BaseRetriever:
        if retriever_type not in cls.retriever_classes:
            raise ValueError(f"Retriever type '{retriever_type}' is not supported.")

        # Check if additional configuration is needed for specific retriever types
        if retriever_type in ["embedding"] and query_embedding_model is None:
            raise ValueError(f"An embedding model name must be provided for '{retriever_type}'.")

        # Initialize the retriever with necessary arguments. Pass **kwargs to support additional parameters.
        if retriever_type in ["embedding"]:
            # Pass embed_model_name and additional kwargs if needed
            return cls.retriever_classes[retriever_type](document_store=document_store,
                                                         embedding_model=query_embedding_model, **kwargs)

        # Initialize the retriever with necessary arguments. Pass **kwargs to support additional parameters.
        if retriever_type in ["dpr", "custom"]:
            # Pass embed_model_name and additional kwargs if needed
            return cls.retriever_classes[retriever_type](document_store=document_store,
                                                         query_embedding_model=query_embedding_model,
                                                         passage_embedding_model=passage_embedding_model,
                                                         **kwargs)
        else:
            # For other retrievers like BM25 that don't require an embedding model name
            return cls.retriever_classes[retriever_type](document_store=document_store, **kwargs)
