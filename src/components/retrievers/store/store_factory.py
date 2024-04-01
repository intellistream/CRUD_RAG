from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore


def create_document_store(store_type: str,
                          **kwargs) -> ElasticsearchDocumentStore | InMemoryDocumentStore:
    if store_type == "elasticsearch":
        return ElasticsearchDocumentStore(**kwargs)
    elif store_type == "inmemory":
        return InMemoryDocumentStore(**kwargs)
    else:
        raise ValueError(f"Unsupported document store type: {store_type}")
