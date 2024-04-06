import logging
import os

from haystack import Pipeline, Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser

# logging.basicConfig(level=logging.DEBUG)
print("Current working directory:", os.getcwd())

if os.path.exists("my_faiss_index.faiss"):
    document_store = FAISSDocumentStore.load("my_faiss_index.faiss")
else:
    document_store = FAISSDocumentStore()
    # Proceed to add documents and generate embeddings

# Check existing document IDs to avoid re-adding documents
existing_doc_ids = {doc.id for doc in document_store.get_all_documents()}
documents = []
documents_dir = 'data/test_docs'
count = 0
for filename in os.listdir(documents_dir):
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

retriever = EmbeddingRetriever(document_store=document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

if document_store.get_embedding_count() < document_store.get_document_count():
    document_store.update_embeddings(retriever, update_existing_embeddings=False)
document_store.save("my_faiss_index.faiss")

rag_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(model_name_or_path="MBZUAI/LaMini-Cerebras-111M",
                         default_prompt_template=rag_prompt)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

output = query_pipeline.run(query="Why did people build Great Pyramid of Giza?")

print(output["answers"][0].answer)
