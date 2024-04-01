from typing import Dict, Any
from haystack.nodes import BaseComponent  # Assuming usage of Haystack's base component for consistency


class BasePrompter(BaseComponent):
    def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate a prompt based on the given context.

        Args:
        context: A dictionary containing data used to generate the prompt.

        Returns:
        A string representing the generated prompt.
        """
        raise NotImplementedError("generate_prompt method must be implemented by subclasses.")
