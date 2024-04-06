import os
from logging import DEBUG

from haystack import Document
from haystack.document_stores import InMemoryDocumentStore, ElasticsearchDocumentStore, FAISSDocumentStore


def create_document_store(store_type: str, documents_dir: str,
                          **kwargs) -> ElasticsearchDocumentStore | InMemoryDocumentStore | FAISSDocumentStore:
    if store_type == "elasticsearch":
        return ElasticsearchDocumentStore(**kwargs)
    elif store_type == "inmemory":
        return InMemoryDocumentStore(**kwargs)
    elif store_type == "faiss":
        # Specify the path where you want to store the faiss_document_store.db
        db_path = os.path.join(documents_dir, 'faiss_document_store.db')
        index_path = os.path.join(documents_dir, 'my_faiss_index.faiss')
        if os.path.exists(index_path):
            return FAISSDocumentStore.load(index_path, **kwargs)
        else:
            return FAISSDocumentStore(sql_url=f"sqlite:///{db_path}", **kwargs)
    else:
        raise ValueError(f"Unsupported document store type: {store_type}")


def initialize_document_store(args):
    # Extract document store type and configuration from args
    store_type = args.store_type
    store_config = vars(args).get("store_config", {})
    print(DEBUG, "Current working directory:", os.getcwd())

    documents = []
    documents_dir = args.docs_path

    # Create the document store using the factory
    document_store = create_document_store(store_type, documents_dir, **store_config)
    # Check existing document IDs to avoid re-adding documents
    existing_doc_ids = {doc.id for doc in document_store.get_all_documents()}
    count = 0
    for filename in os.listdir(documents_dir):
        if filename.endswith('.db') or filename.endswith('.faiss') or filename.endswith('.json'):
            # Skip files with these extensions
            continue
        file_path = os.path.join(documents_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                for index, line in enumerate(file.readlines()):
                    line = line.strip()
                    if line:
                        # Create a unique ID for each document
                        doc_id = f"{filename}-{index}"
                        # Skip if document already exists
                        if doc_id in existing_doc_ids:
                            continue
                        document = Document(content=line, id=doc_id)
                        documents.append(document)
                count += 1
                if count == 1:
                    break
    if documents:
        document_store.write_documents(documents, duplicate_documents='skip')
    else:
        print("No new documents to add.")
    return document_store
