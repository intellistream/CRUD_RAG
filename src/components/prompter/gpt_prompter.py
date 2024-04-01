from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate, AnswerParser
import openai
from src.configs.config import GPT_api_key, GPT_api_base


class GPTPrompter(PromptNode):
    def __init__(self, model_name_or_path: str, temperature: float = 1.0, max_new_tokens: int = 1024, top_p: float = 0.9, **kwargs):
        self.api_key = GPT_api_key
        self.api_base = GPT_api_base if GPT_api_base.strip() else None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        # Simulating a PromptTemplate for direct OpenAI API call
        prompt_template = PromptTemplate(
            prompt="Question: {query}\nAnswer:",
            output_parser=AnswerParser()
        )

        # Although not directly using PromptModel, define it for consistency with LFQAPrompter style
        prompt_model = PromptModel(model_name_or_path=model_name_or_path, **kwargs)

        super().__init__(model=prompt_model, default_prompt_template=prompt_template)

    def _query_openai_api(self, query: str):
        if self.api_base:
            openai.api_base = self.api_base
        openai.api_key = self.api_key
        response = openai.Completion.create(
            model=self.model.model_name_or_path,
            prompt=f"Question: {query}\nAnswer:",
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p
        )
        return response.choices[0].text.strip()

    def run(self, query: str, documents=None):
        # Directly invoke OpenAI's API for GPT, bypassing the usual model invocation
        answer = self._query_openai_api(query)
        return {"answers": [{"answer": answer}]}
