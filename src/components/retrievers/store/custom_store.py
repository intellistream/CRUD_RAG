import pandas as pd
import torch
from haystack import Document
from haystack.document_stores import BaseDocumentStore


class RANIADocumentStore(BaseDocumentStore):
    """
    Document Store utilizing RANIA for efficient indexing and retrieval.

    Assumes RANIA library is loaded and provides specific functions for indexing and retrieval.
    """

    def __init__(self, rania_lib_path, index_name, collection_path, encoder_model, tokenizer, config=None):
        super().__init__()
        self.index_name = index_name
        self.collection_path = collection_path
        self.encoder_model = encoder_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.config = config or {}

        # Load the RANIA library
        torch.ops.load_library(rania_lib_path)

        # Initialize RANIA with the specific index configuration
        self._init_rania()

        # Load documents from a CSV/TSV file into a DataFrame
        self.docs = pd.read_csv(
            collection_path, sep="\t" if collection_path.endswith(".tsv") else ",", header=None
        )

        # Load documents into RANIA
        self._load_documents_into_rania()

    def _init_rania(self):
        """Initialize the RANIA index."""
        torch.ops.RANIA.index_create(self.rania_name, 'flatAMMIPObj') # This is unclear to me. @xianzhi
        torch.ops.RANIA.index_editCfgI64(self.index_name, 'vecDim', 768)  # Example config, adjust as needed
        torch.ops.RANIA.index_init(self.index_name)

    def _load_documents_into_rania(self):
        """Encode and load documents from the DataFrame into RANIA."""
        for _, row in self.docs.iterrows():
            doc_id, content = row[0], row[1]  # Assuming the first column is 'doc_id' and the second is 'content'
            self._add_document_to_rania(doc_id, content)

    def _add_document_to_rania(self, doc_id, content):
        """
        Encode a document using the provided encoder and tokenizer,
        then add it to the RANIA index.
        """
        inputs = self.tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.encoder_model(**inputs.to(self.encoder_model.device))
        # Assuming `outputs.pooler_output` is the desired embedding vector
        embeddings = outputs.pooler_output
        # Insert document into RANIA. Adjust this to your RANIA's API.
        torch.ops.RANIA.index_insertString(self.index_name, embeddings, [content])

    def _load_index(self):
        """Load PLAID index from the paths given to the class and initialize a Searcher object."""
        with Run().context(
                RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            self.store = Searcher(
                index="", collection=self.collection_path, checkpoint=self.checkpoint_path
            )

        logger.info("Loaded PLAIDDocumentStore index")

    def _create_index(self):
        """Generate a PLAID index from a given ColBERT checkpoint.

        Given a checkpoint and a collection of documents, an Indexer object will be created.
        The index will then be generated, written to disk at `index_path` and finally it
        will be loaded.
        """

        with Run().context(
                RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                kmeans_niters=self.kmeans_niters,
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index("", collection=self.collection_path, overwrite=True)

        logger.info("Created PLAIDDocumentStore Index.")

    def write_documents(self, dataset, batch_size=1):
        raise NotImplementedError

    def get_all_documents(self):
        raise NotImplementedError

    def get_all_documents_generator(self):
        raise NotImplementedError

    def delete_index(self):
        raise NotImplementedError

    def get_all_labels(self):
        raise NotImplementedError

    def query_by_embedding(self):
        raise NotImplementedError

    def get_label_count(self):
        raise NotImplementedError

    def write_labels(self):
        raise NotImplementedError

    def delete_documents(self):
        raise NotImplementedError

    def delete_labels(self):
        raise NotImplementedError

    def _create_document_field_map(self):
        raise NotImplementedError

    def get_documents_by_id(self):
        raise NotImplementedError

    def get_document_by_id(self):
        raise NotImplementedError

    def update_document_meta(self):
        raise NotImplementedError
    def get_document_count(self):
        """
        Returns the number of docs in the collection.
        """
        return len(self.docs)

    def query(self, query_text, top_k=10):
        """Query the RANIA index and return the top K matching documents."""
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.encoder_model(**inputs.to(self.encoder_model.device))
        query_embeddings = outputs.pooler_output
        search_results = torch.ops.RANIA.index_searchString(self.index_name, query_embeddings, top_k)

        documents = []
        for doc_id in search_results[0]:
            content = self.docs.iloc[int(doc_id)][1]
            documents.append(Document(content=content, id=str(doc_id)))
        return documents

    def query_batch(self, query_strs: List[str], top_k=10) -> List[List[Document]]:
        """
        Query batch the Colbert v2 + Plaid store.

        Returns: lists of lists of Haystack documents.
        """

        query = self.store.search_all({i: s for i, s in enumerate(query_strs)}, k=top_k)
        documents = []

        for result in query.data.values():
            s_docs = [
                Document.from_dict(
                    {
                        "content": self.docs.iloc[_id][1],
                        "id": _id,
                        "score": score,
                        "meta": {"title": self.docs.iloc[_id][2] if self.titles else None},
                    }
                )
                for _id, _, score in result
            ]
            documents.append(s_docs)

        # for docs in documents:
        #     self._normalize_scores(docs)

        return documents
