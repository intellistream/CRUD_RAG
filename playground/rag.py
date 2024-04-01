from fastrag.prompters.invocation_layers.llama_cpp import LlamaCPPInvocationLayer
from haystack import Pipeline
from haystack.nodes.prompt import PromptNode
from haystack.nodes import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.nodes import AnswerParser
from haystack.nodes.ranker import SentenceTransformersRanker
from haystack.nodes.retriever import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document

from retriever import RANIARetriever

retriever = RANIARetriever(top_k=1, custom_rania_name='warthunder')
reranker = SentenceTransformersRanker(
    batch_size=32,
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=1,
    use_gpu=False
)
AParser = AnswerParser()
LFQA = PromptTemplate(
    prompt="""{join(documents)}
Question: {query}
Answer: """,
    output_parser=AParser
)
PrompterModel = PromptModel(
    model_name_or_path="models/marcoroni-7b-v3.Q4_K_M.gguf",
    invocation_layer_class=LlamaCPPInvocationLayer,
    model_kwargs=dict(
        max_new_tokens=50
    )
)
Prompter = PromptNode(
    model_name_or_path=PrompterModel,
    default_prompt_template=LFQA
)
pipe = Pipeline()

pipe.add_node(component=retriever, name='Retriever', inputs=["Query"])
pipe.add_node(component=reranker, name='Reranker', inputs=["Retriever"])
pipe.add_node(component=Prompter, name='Prompter', inputs=["Reranker"])
