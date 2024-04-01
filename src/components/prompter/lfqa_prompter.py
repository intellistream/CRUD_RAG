import torch
from haystack.nodes import PromptNode, BaseComponent, AnswerParser
from haystack.nodes.prompt.prompt_template import PromptTemplate

class LFQAPrompter(PromptNode):
    def __init__(self, model_name_or_path: str, **kwargs):
        # Configure the AnswerParser as needed for LFQA
        answer_parser = AnswerParser()

        # Set up the LFQA PromptTemplate

        # For Summarization
        lfqa_template = PromptTemplate(
            prompt="""Given the context please summarize the event. Context:{join(documents)}
                    Summarize this event: {query}
                    Summary: """,
        )
        # For Question-Answering
        # lfqa_template = PromptTemplate(
        #     prompt="""{join(documents, attribute='text')}
        #             Question: {query}
        #             Answer: """,
        #     output_parser=answer_parser
        # )

        # Initialize the PromptNode with the model and LFQA template
        super().__init__(model_name_or_path=model_name_or_path, default_prompt_template=lfqa_template, model_kwargs={
            "model_max_length": 2048, "torch_dtype": torch.bfloat16})
