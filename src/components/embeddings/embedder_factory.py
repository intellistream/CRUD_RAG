from typing import Union

from src.components.embeddings import HaystackEmbedder, CustomEmbedder


class EmbedderFactory:
    @staticmethod
    def get_embedder(type: str, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                     device: str = 'cuda') -> Union[HaystackEmbedder, CustomEmbedder]:
        """
        Factory method to get the specified embedder.

        Args:
            type (str): Type of the embedder to create ("haystack" or "custom").
            model_name_or_path (str): The model name or path for the embedder.
            device (str): Computation device ('cuda' or 'cpu'), applicable to the custom embedder.

        Returns:
            Union[HaystackEmbedder, CustomEmbedder]: An instance of the specified embedder.
        """
        if type == "haystack":
            return HaystackEmbedder(model_name=model_name_or_path)
        elif type == "custom":
            return CustomEmbedder(model_name_or_path=model_name_or_path, device=device)
        else:
            raise ValueError("Unsupported embedder type. Choose 'haystack' or 'custom'.")


# Example usage
# For Haystack embedder
haystack_embedder = EmbedderFactory.get_embedder("haystack",
                                                 model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

# For custom embedder
custom_embedder = EmbedderFactory.get_embedder("custom", model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                                               device='cuda')

# Usage
query = ["What is the capital of France?", "Explain quantum entanglement."]
print(haystack_embedder.embed_queries(query))
print(custom_embedder.embed_queries(query))
