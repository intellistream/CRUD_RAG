import os

from haystack import Pipeline

from src.components.prompter.prompter_factory import PrompterFactory
from src.components.reranker.reranker_factory import RerankerFactory
from src.components.retrievers.retriever_factory import RetrieverFactory
from src.components.retrievers.store.store_factory import initialize_document_store


def basic_pipeline(args):
    """
    Initializes a basic pipeline with a configurable document store.

    Args:
    args: Command-line arguments or any configuration object that includes
          the document store type and additional configurations.

    Returns:
    A Haystack Pipeline with the specified document store.
    """
    document_store = initialize_document_store(args)

    pipeline = Pipeline()
    # Add a retriever
    # Ensure the retriever is initialized before updating embeddings
    retriever = RetrieverFactory.get_retriever(retriever_type=args.retriever_type,
                                               document_store=document_store,
                                               query_embedding_model=args.query_embedding_model,
                                               passage_embedding_model=args.passage_embedding_model)

    if document_store.get_embedding_count() < document_store.get_document_count():
        document_store.update_embeddings(retriever, update_existing_embeddings=False)

    index_path = os.path.join(args.docs_path, 'my_faiss_index.faiss')
    document_store.save(index_path)

    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    # Based on the retriever's configuration, decide if a reranker is needed
    if args.need_reranker:
        reranker = RerankerFactory.get_reranker(reranker_type=args.reranker_type, batch_size=32,
                                                model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                                top_k=1,
                                                use_gpu=False)
        pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
        last_component = "Reranker"
    else:
        last_component = "Retriever"

    # Finally, add a prompter or generator
    prompter = PrompterFactory.create_prompter(prompter_type=args.prompter_type,
                                               model_name_or_path=args.model_name_or_path)
    pipeline.add_node(component=prompter, name="Prompter", inputs=[last_component])
    return pipeline


def advanced_pipeline(args):
    pass


class PipelineFactory:
    @staticmethod
    def create_pipeline(args):
        pipeline_type = args.pipeline_type
        if pipeline_type == "basic":
            # Pass args or its specific properties as needed
            return basic_pipeline(args)
        elif pipeline_type == "advanced":
            # Pass args or its specific properties as needed
            return advanced_pipeline(args)
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
