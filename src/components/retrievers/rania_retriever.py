from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
import torch
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import DensePassageRetriever


class RANIARetriever(DensePassageRetriever):
    def __init__(
            self,
            document_store: Optional[BaseDocumentStore] = None,
            query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base",
            model_version: Optional[str] = None,
            max_seq_len_query: int = 64,
            max_seq_len_passage: int = 256,
            top_k: int = 10,
            use_gpu: bool = True,
            batch_size: int = 16,
            embed_title: bool = True,
            use_fast_tokenizers: bool = True,
            similarity_function: str = "dot_product",
            global_loss_buffer_size: int = 150000,
            progress_bar: bool = True,
            devices: Optional[List[Union[str, "torch.device"]]] = None,
            use_auth_token: Optional[Union[str, bool]] = None,
            scale_score: bool = True,
    ):
        # Initialize the superclass with all provided arguments
        super().__init__(
            document_store=document_store,
            query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
            model_version=model_version,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            top_k=top_k,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers,
            similarity_function=similarity_function,
            global_loss_buffer_size=global_loss_buffer_size,
            progress_bar=progress_bar,
            devices=devices,
            use_auth_token=use_auth_token,
            scale_score=scale_score,
        )

    def insert_documents(self, documents: List[str], batch_size: int = 16):
        """
        Insert a batch of documents into the document store.

        Args:
            documents (List[str]): The documents to insert.
            batch_size (int): The number of documents to insert in a single batch.
        """
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            embeddings = self.embed_documents(batch_docs)
            self.document_store.write_documents(batch_docs, embeddings)
        print(f"Inserted {len(documents)} documents into the document store.")

    def delete_documents(self, document_ids: List[str]):
        """
        Delete a batch of documents from the document store based on document IDs.

        Args:
            document_ids (List[str]): The IDs of the documents to delete.
        """
        self.document_store.delete_documents(document_ids=document_ids)
        print(f"Deleted {len(document_ids)} documents from the document store.")

    def search(self, query: str, top_k: int = 10, filter: Optional[dict] = None):
        """
        Perform a search in the document store using the query embeddings.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to retrieve.
            filter (dict, optional): Any filters to apply to the search.

        Returns:
            List[Document]: The top_k documents from the search results.
        """
        query_embedding = self.embed_queries([query])[0]
        results = self.document_store.query(query_embedding, top_k=top_k, filter=filter)
        return results

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Override the DensePassageRetriever's embed_documents method if needed
        to customize the embedding process for documents.
        """
        # Example: Customize embedding logic or preprocess documents
        return super().embed_documents(documents)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Override the DensePassageRetriever's embed_queries method if needed
        to customize the embedding process for queries.
        """
        # Example: Customize embedding logic or preprocess queries
        return super().embed_queries(queries)

    # Add more custom methods or overrides as needed for your application.
