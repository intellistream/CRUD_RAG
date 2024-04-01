from src.components.prompter.gpt_prompter import GPTPrompter
from src.components.prompter.lfqa_prompter import LFQAPrompter


class PrompterFactory:
    @staticmethod
    def create_prompter(prompter_type: str, **kwargs):
        if prompter_type == "lfqa":
            return LFQAPrompter(**kwargs)
        elif prompter_type == "gpt":
            return GPTPrompter(**kwargs)
        # Add other elif branches for different prompter types as needed
        else:
            raise ValueError(f"Unsupported prompter type: {prompter_type}")
