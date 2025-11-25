# src/contextchatbot/chatbot2.py

import pandas as pd
import tiktoken
import openai
import os
from typing import Tuple

class ContextualChatbot:
    """
    A contextual chatbot agent that uses a local CSV for Retrieval-Augmented Generation (RAG).
    """
    
    # Constants should be class or instance attributes, not global variables
    MAX_TOKEN_COUNT = 4097
    MAX_RESPONSE_LENGTH = 150
    COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"

    def __init__(self, data_path: str = 'data/2023_fashion_trends.csv'):
        
        # API Key is loaded via os.getenv in the main CLI, but we ensure it's set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not found. Check your .env file.")
            
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Load CSV data from the expected relative path
        try:
            df = pd.read_csv(data_path)
            self.context_data = df["Trends"].values
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # The prompt template is also stored as an attribute
        self.prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""


    def _create_prompt(self, question: str) -> Tuple[int, str]:
        """Creates the prompt by filling the context until max token count is reached."""
        
        current_token_count = len(self.tokenizer.encode(self.prompt_template)) + \
                                len(self.tokenizer.encode(question))
        
        context = []
        # Calculate the maximum tokens allowed for context
        max_prompt_tokens = self.MAX_TOKEN_COUNT - self.MAX_RESPONSE_LENGTH

        for text in self.context_data:

            text_token_count = len(self.tokenizer.encode(text))
            
            # Add the row of text to the list if we haven't exceeded the max
            if current_token_count + text_token_count <= max_prompt_tokens:
                context.append(text)
                current_token_count += text_token_count
            else:
                break
        
        return current_token_count, self.prompt_template.format("\n\n###\n\n".join(context), question)


    def answer_question(self, question: str) -> Tuple[int,str, str, str]:
        """
        Queries the OpenAI API with a context-augmented prompt.
        Returns: (tokens used, answer text, full prompt sent)
        """
        
        num_tokens, prompt = self._create_prompt(question)

        try:
            #Add context from dataset
            response_without_context = openai.Completion.create(
                model=self.COMPLETION_MODEL_NAME,
                prompt=question,
                max_tokens=self.MAX_RESPONSE_LENGTH
            )

            #Add context from dataset
            response_with_context = openai.Completion.create(
                model=self.COMPLETION_MODEL_NAME,
                prompt=prompt,
                max_tokens=self.MAX_RESPONSE_LENGTH
            )

            return  num_tokens, question, response_without_context["choices"][0]["text"].strip(), response_with_context["choices"][0]["text"].strip()
        except Exception as e:
            # You should use a logger, but print is fine for this example
            print(f"OpenAI API Error: {e}") 
            return 0, "I apologize, an error occurred during the API call.", ""