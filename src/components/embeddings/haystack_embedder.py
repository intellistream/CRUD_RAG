from haystack.nodes import EmbeddingRetriever
import torch

class HaystackEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_format: str = "sentence_transformers"):
        """
        Initializes the embedder with the specified model.

        Args:
            model_name (str): The name or path of the embedding model.
            model_format (str): The format or library of the model (e.g., "sentence_transformers").
        """
        # Assuming the document_store is managed elsewhere and not directly needed for embedding alone
        self.embedder = EmbeddingRetriever(
            document_store=None,  # Specify if you have a document store
            embedding_model=model_name,
            model_format=model_format,
        )

    def embed_queries(self, queries: list) -> torch.Tensor:
        """
        Embeds a list of queries.

        Args:
            queries (list): Queries to embed.

        Returns:
            torch.Tensor: The embeddings as a tensor.
        """
        embeddings = self.embedder.embed_queries(texts=queries)
        return embeddings

    def embed_documents(self, documents: list) -> torch.Tensor:
        """
        Embeds a list of documents.

        Args:
            documents (list): Documents to embed.

        Returns:
            torch.Tensor: The embeddings as a tensor.
        """
        embeddings = self.embedder.embed(texts=documents)
        return embeddings

# Example usage
# embedder = HaystackEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# query_embeddings = embedder.embed_queries(["What is the capital of France?", "Explain quantum entanglement."])
# document_embeddings = embedder.embed_documents(["Paris is the capital of France.", "Quantum entanglement is a physical phenomenon."])
#
# print(query_embeddings.shape, document_embeddings.shape)
