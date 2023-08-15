import time

import numpy as np
import openai
import yaml
from transformers import GPT2Tokenizer

MAX_OPENAI_LEN = 4096


class OpenAIClient:
    def __init__(self):
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_response_tokens = self.config['OpenAIClient']['max_response_tokens']
        self.temperature = self.config['OpenAIClient']["temperature"]
        openai.api_key = "02e3dbabaf334ccb959cbeadbd3f99c3" #TODO replace with .env file
        openai.api_base = "https://llm-x-gpt.openai.azure.com/"#TODO replace with .env file
        openai.api_type = 'azure' #TODO move to config?
        openai.api_version = '2023-05-15'  # this may change in the future
        self.instruct_deployment_name = self.config['OpenAIClient']['instruct_deployment_name']
        self.embedding_deployment_name = self.config['OpenAIClient']['embedding_deployment_name']

    def generate_response(self, prompt: str) -> str:
        time.sleep(0.5) # ToDo: This avoid rate limit error. Remove when rate limit is fixed.
        prompt = self._truncate_prompt(prompt=prompt, truncate_end=True)
        try:
            response = openai.Completion.create(engine=self.instruct_deployment_name,
                                                prompt=prompt,
                                                max_tokens=self.max_response_tokens,
                                                temperature=self.temperature)
            text = response['choices'][0]['text']
        except Exception as e:
            if response is not None and "error" in response and "message" in response["error"]:
                print(response["error"]["message"])
            else:
                print(e)
            raise e

        return text

    def get_openai_embeddings(self, input: str) -> np.array:
        time.sleep(0.5) # ToDo: Remove.
        try:
            response = openai.Embedding.create(
                input=input,
                engine=self.embedding_deployment_name
            )
            embeddings = response['data'][0]['embedding']
        except Exception as e:
            if response is not None and "error" in response and "message" in response["error"]:
                print(response["error"]["message"])
            else:
                print(e)
            raise e
        return np.array(embeddings)

    def _truncate_prompt(self, prompt: str, truncate_end: bool) -> str:
        """
        We have a limit to the prompt + response length
        If we pass this we must truncate the prompt
        """
        prompt_tokens = self.tokenizer(prompt)['input_ids']
        prompt_len = len(prompt_tokens)
        max_prompt_tokens = MAX_OPENAI_LEN - self.max_response_tokens
        if prompt_len > max_prompt_tokens:
            # ToDo: Add logging the truncated prompt.
            if truncate_end:
                prompt_tokens = prompt_tokens[:max_prompt_tokens]
            else:
                prompt_tokens = prompt_tokens[-max_prompt_tokens:]

            truncated_prompt = self.tokenizer.decode(prompt_tokens)
            return truncated_prompt
        else:
            return prompt