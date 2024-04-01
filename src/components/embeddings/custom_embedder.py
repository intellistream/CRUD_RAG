import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class CustomEmbedder:
    def __init__(self, model_name_or_path: str, device: str = 'cuda'):
        """
        Initializes the custom embedder with a specified transformer model.

        Args:
            model_name_or_path (str): Path or identifier for the pretrained model to use.
            device (str): The device to use for computation ('cuda' or 'cpu').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.model.eval()  # Set the model to evaluation mode

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to encode.
            batch_size (int): Size of batches for processing texts.

        Returns:
            torch.Tensor: A tensor containing embeddings for the input texts.
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        n_seq = len(encoded_input['input_ids'])
        shape_out = (n_seq, self.model.config.hidden_size)
        embeddings_out = torch.zeros(shape_out).to(self.device)

        for batch_start in range(0, n_seq, batch_size):
            batch_end = batch_start + batch_size
            batch_input_ids = encoded_input['input_ids'][batch_start:batch_end].to(self.device)
            batch_attention_mask = encoded_input['attention_mask'][batch_start:batch_end].to(self.device)

            with torch.no_grad():
                batch_outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                batch_embeddings = batch_outputs.pooler_output  # Adjust based on the model
                embeddings_out[batch_start:batch_end] = batch_embeddings

        return embeddings_out.cpu()

    # Wrapper methods for ease of use
    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Embeds a list of queries.

        Args:
            queries (List[str]): Queries to embed.

        Returns:
            torch.Tensor: The embeddings.
        """
        return self.generate_embeddings(queries)

    def embed_documents(self, documents: List[str]) -> torch.Tensor:
        """
        Embeds a list of documents.

        Args:
            documents (List[str]): Documents to embed.

        Returns:
            torch.Tensor: The embeddings.
        """
        return self.generate_embeddings(documents)

# Example usage
# model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
# custom_embedder = CustomEmbedder(model_name_or_path, device='cuda')
#
# query_embeddings = custom_embedder.embed_queries(["What is the capital of France?", "Explain quantum entanglement."])
# document_embeddings = custom_embedder.embed_documents(["Paris is the capital of France.", "Quantum entanglement is a physical phenomenon."])
#
# print(query_embeddings.shape, document_embeddings.shape)
