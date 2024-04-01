from haystack import Pipeline
from haystack.nodes.prompt import PromptNode
from haystack.nodes import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.nodes import AnswerParser
from haystack.nodes.ranker import SentenceTransformersRanker
from haystack.nodes.retriever import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import BatchEncoding
import torch
import time
import os, shutil
import gzip
from datasets import load_dataset, Dataset
# the files used/created by pickle are temporary and don't pose any security issue
import pickle  # nosec
import random
import numpy as np
import numpy.typing as npt
import nltk
import sys
from typing import Optional, Union
from typing import Dict, List, Optional, Union, Any

import logging
from collections import OrderedDict, namedtuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from haystack.schema import Document
from haystack.document_stores.base import BaseDocumentStore, FilterType
from haystack.document_stores import KeywordDocumentStore
from haystack.nodes.retriever import BaseRetriever
from haystack.errors import DocumentStoreError


class RANIARetriever(BaseRetriever):
    def __init__(
            self,
            document_store: Optional[KeywordDocumentStore] = None,
            top_k: int = 10,
            all_terms_must_match: bool = False,
            custom_query: Optional[str] = None,
            custom_rania_name: Optional[str] = 'aknn0',
            scale_score: bool = True,
    ):
        super().__init__()
        self.document_store: Optional[KeywordDocumentStore] = document_store
        self.top_k = top_k
        self.custom_query = custom_query
        self.all_terms_must_match = all_terms_must_match
        self.scale_score = scale_score
        self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.ctx_encoder = self.ctx_encoder.to(self.device)
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.q_encoder = self.q_encoder.to(self.device)
        """Loading RANIA

        """
        torch.ops.load_library("../../../libRANIA.so")
        self.rania_name = custom_rania_name
        # gen the input tensor
        torch.ops.RANIA.index_create(self.rania_name, 'flatAMMIPObj')
        torch.ops.RANIA.index_editCfgI64(self.rania_name, 'vecDim', 768)
        torch.ops.RANIA.index_init(self.rania_name)

    def retrieve(
            self,
            query: str,
            filters: Optional[FilterType] = None,
            top_k: Optional[int] = None,
            all_terms_must_match: Optional[bool] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        """
        embedQ, listQ = self.encodeQuery(query)
        q0 = embedQ[0:1, :]
        ru = torch.ops.RANIA.index_searchString(self.rania_name, q0, self.top_k)
        strList = ru[0]
        documents = [Document(content=item) for item in strList]
        return documents

    def retrieve_batch(
            self,
            queries: List[str],
            filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
            top_k: Optional[int] = None,
            all_terms_must_match: Optional[bool] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            batch_size: Optional[int] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        documents = [self.retrieve(i) for i in queries]
        return documents

    def generate_embeddings(self, model: Union[DPRContextEncoder, DPRQuestionEncoder], encoded_input: BatchEncoding,
                            dim: int,
                            batch_size: int, device: str) -> torch.tensor:
        n_seq = len(encoded_input['input_ids'])
        shapeOut = (n_seq, dim)
        token_embeddings_out = torch.zeros(shapeOut)

        print('Doing inference for', n_seq, 'sequences.')

        model.eval()

        num_batches = int(np.ceil(float(n_seq) / batch_size))
        batch_print = 100
        if device != "cpu":
            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)
            start1.record()
        with torch.no_grad():
            for batch in range(num_batches):

                batch_init = batch * batch_size
                batch_end = np.min([batch_init + batch_size, n_seq])

                token_embeddings = model(encoded_input['input_ids'][batch_init:batch_end].to(device),
                                         encoded_input['attention_mask'][batch_init:batch_end].to(device))
                token_embeddings_out[batch_init:batch_end, :] = token_embeddings.pooler_output.cpu()
                if not (batch % batch_print):
                    print('Doing inference for batch', batch, 'of', num_batches)
        if device != "cpu":
            end1.record()
            torch.cuda.synchronize()
            print(f'Inference for {n_seq}, sequences took {(start1.elapsed_time(end1) / 1000):.2f} s')

        return token_embeddings_out

    def tokenize_texts(self, ctx_tokenizer: AutoTokenizer, texts: list, max_length: Optional[int] = None,
                       doc_stride: Optional[int] = None,
                       text_type: Optional[str] = "context", save_sentences: Optional[bool] = False, \
                       fname_sentences: Optional[str] = None) -> BatchEncoding:
        if text_type == "context":
            if max_length == None:
                max_length = 2048
                print("Setting max_length to", max_length)
            if doc_stride == None:
                doc_stride = int(max_length / 2)
                print("Setting doc_stride to", doc_stride)

        start = time.time()
        if text_type == "context":
            encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, max_length=max_length, \
                                           return_overflowing_tokens=True, \
                                           stride=doc_stride, return_tensors="pt")
        elif text_type == "query":
            encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        end = time.time()
        delta_time = end - start
        print(f'Tokenization for {len(texts)}, contexts took {delta_time:.2f} s')

        n_seq = len(encoded_inputs['input_ids'])
        if save_sentences:
            if fname_sentences is not None:
                # Code to generate sentences from tokens
                sentences = []
                for i in range(n_seq):
                    if not (i % 100000):
                        print('Processing sentence', i, 'of', n_seq)
                    sentences += [' '.join(encoded_inputs.tokens(i))]

                with open(fname_sentences, 'wb') as f:
                    pickle.dump(sentences, f)
                del sentences
            else:
                raise BaseException(
                    'tokenize_texts: The filename where the original sentences will be saved was not specified.')

        return encoded_inputs

    def encodeContext(self, ctx: str, batchSize: Optional[int] = 64):
        encoded_contexts = self.tokenize_texts(self.ctx_tokenizer, ctx, text_type="context").to(self.device)
        embeddings_batch = self.generate_embeddings(self.ctx_encoder, encoded_contexts, 768, batchSize,
                                                    self.device)
        # Define the string to initialize each element
        initial_string = ctx

        # Create the list using a list comprehension
        list_of_strings = [initial_string for _ in range(embeddings_batch.size(0))]
        return embeddings_batch, list_of_strings

    def encodeQuery(self, qtx: str, batchSize: Optional[int] = 64):
        text_type = "query"
        encoded_quries = self.tokenize_texts(self.q_tokenizer, qtx, text_type=text_type).to(self.device)
        embeddings_batch = self.generate_embeddings(self.q_encoder, encoded_quries, 768, batchSize,
                                                    self.device)
        initial_string = qtx

        # Create the list using a list comprehension
        list_of_strings = [initial_string for _ in range(embeddings_batch.size(0))]
        return embeddings_batch, list_of_strings

    def insertContext(self, ctx: str):
        embeddings_batch, list_of_strings = self.encodeContext(ctx)
        return torch.ops.RANIA.index_insertString(self.rania_name, embeddings_batch, list_of_strings)

    def deleteContext(self, ctx: str):
        embeddings_batch, list_of_strings = self.encodeContext(ctx)
        return torch.ops.RANIA.index_deleteString(self.rania_name, embeddings_batch, 1)

    def deleteQuery(self, qtx: str, k=1):
        embeddings_batch, list_of_strings = self.encodeQuery(qtx)
        return torch.ops.RANIA.index_deleteString(self.rania_name, embeddings_batch, k)