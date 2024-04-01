import os
from haystack import Pipeline, Document

from src.components.prompter.prompter_factory import PrompterFactory
from src.components.reranker.reranker_factory import RerankerFactory
from src.components.retrievers.retriever_factory import RetrieverFactory
from src.components.retrievers.store.store_factory import create_document_store


def basic_pipeline(args):
    """
    Initializes a basic pipeline with a configurable document store.

    Args:
    args: Command-line arguments or any configuration object that includes
          the document store type and additional configurations.

    Returns:
    A Haystack Pipeline with the specified document store.
    """
    # Extract document store type and configuration from args
    store_type = args.store_type
    store_config = vars(args).get("store_config", {})

    # Create the document store using the factory
    document_store = create_document_store(store_type, **store_config)

    documents = []
    documents_dir = args.docs_path
    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                document = Document(content=content)
                documents.append(document)
    document_store.write_documents(documents)

    pipeline = Pipeline()
    # Add a retriever
    # Use the RetrieverFactory to get the retriever instance
    retriever = RetrieverFactory.get_retriever(retriever_type=args.retriever_type,
                                               document_store=document_store,
                                               query_embedding_model=args.query_embedding_model,
                                               passage_embedding_model=args.passage_embedding_model)
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
