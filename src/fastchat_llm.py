import json


from langchain.chat_models import AzureChatOpenAI

# General constants
DEFAULT_TEMPERATURE = 0.01

# GPT constants
PROMPT_TEMPLATE = '{prompt}'
PROMPT_FIELD = 'prompt'
GPT35 = AzureChatOpenAI(
    temperature=DEFAULT_TEMPERATURE,
    openai_api_key='e74153596f8047c7ba7e4a07ff108138',
    openai_api_base='https://llm-x-gpt.openai.azure.com/',
    deployment_name='LLM-X-GPT35-TURBO',
    openai_api_version='2023-03-15-preview',
)
GPT4 = AzureChatOpenAI(
    temperature=DEFAULT_TEMPERATURE,
    openai_api_key='74c1368e780d48698a9979c474abcc8a',
    openai_api_base='https://llmx-gpt-canada-east.openai.azure.com/',
    deployment_name='LLM-X-GPT-4',
    openai_api_version='2023-03-15-preview',
)

# FastChat constants
FASTCHAT_LLM_NAME = 'fastchat_llm'
LLM_DEFAULT_IDENTIFYING_PARAMETERS = {}
TEXT_RESPONSE_DEFAULT_VALUE = ''
RESPONSE_CHUNK_DELIMITER = b"\0"
RESPONSE_CHUNK_TEXT_FIELD = 'generated_text'



from typing import Any, Mapping
import requests
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from requests.models import Response

from src.request_payload import FastChatRequestPayload

FASTCHAT_URL = "TODO"
DEFAULT_TEMPRATURE = 0.2

RESPONSE_CHUNK_DELIMIETER = b"\0"
TEXT_FIELD_NAME_IN_RESPONSE_CHUNK = "text"
TEXT_RESPONSE_DEFAULT_VALUE = ''
FASTCHAT_LLM_NAME = "fastchat_llm"
LLM_DEFAULT_IDENTIFYING_PARAMETERS = {}


class FastchatLLM(LLM):
    _fastchat_model: str = PrivateAttr(default_factory=str)
    _temprature: str = PrivateAttr(default_factory=str)

    def __init__(self, fastchat_model: str, temperature: float = DEFAULT_TEMPRATURE):
        super().__init__()
        self._fastchat_model = fastchat_model
        self._temperature = temperature


    @property
    def _llm_type(self) -> str:
        return FASTCHAT_LLM_NAME

    @property
    def _identyfing_params(self) -> Mapping[str, Any]:
        return LLM_DEFAULT_IDENTIFYING_PARAMETERS

    def _call(self, prompt: str, **kwargs: Any) -> str:
        model_request_payload = self._build_model_request_payload(prompt)

        response = requests.post(FASTCHAT_URL, json=model_request_payload.dict())
        response.raise_for_status()

        content = _get_model_response_from_stream(response)

        return content

    def _build_model_request_payload(self, prompt: str) -> FastChatRequestPayload:
        payload = FastChatRequestPayload(model=self._fastchat_model, prompt=prompt, temprature=self._temprature)
        return payload


def _get_model_response_from_stream(response: Response) -> str:
    text_response = TEXT_RESPONSE_DEFAULT_VALUE
    for chunk in response.iter_lines(decode_unicode=False, delimiter=RESPONSE_CHUNK_DELIMIETER):
        if chunk:
            data = json.loads(chunk.decode())
            chunk_text_response = data[TEXT_FIELD_NAME_IN_RESPONSE_CHUNK]
            text_response = chunk_text_response

    return text_response

# ------------------------------ #

import json
import requests
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from requests.models import Response
from typing import Any, Mapping


class FastChatLLM(LLM):
    _model_host: str = PrivateAttr(default_factory=str)
    _fastchat_url: str = PrivateAttr(default_factory=str)
    _base_model_name: str = PrivateAttr(default_factory=str)
    _temperature: float = PrivateAttr(default_factory=float)

    def __init__(
            self,
            model_host: str,
            fastchat_url: str,
            base_model_name: str,
            temperature: float = DEFAULT_TEMPERATURE
    ):
        super().__init__()
        self._model_host = model_host
        self._fastchat_url = fastchat_url
        self._base_model_name = base_model_name
        self._temperature = temperature

    @property
    def _llm_type(self) -> str:
        return FASTCHAT_LLM_NAME

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return LLM_DEFAULT_IDENTIFYING_PARAMETERS

    def _call(self, prompt: str, **kwargs: Any) -> str:
        request_headers = self._build_request_headers()
        # model_request_payload = self._build_model_request_payload(prompt)
        request_body = self._build_request_body(prompt)

        response = requests.post(self._fastchat_url, headers=request_headers, json=request_body)
        response.raise_for_status()

        content = _get_model_response_from_stream(response)

        return content

    def _build_request_headers(self) -> Mapping[str, str]:
        request_headers = {'User-Agent': 'FastChat Client', 'Host': self._model_host}
        return request_headers

    def _build_request_body(self, prompt: str) -> dict:
        request_body = {
            "inputs": prompt,
            "parameters": {
                "best_of": 1,
                "decoder_input_details": True,
                "details": True,
                "do_sample": True,
                "max_new_tokens": 20,
                "repetition_penalty": 1.03,
                "return_full_text": False,
                "seed": None,
                "stop": [
                    "photographer"
                ],
                "temperature": DEFAULT_TEMPERATURE,
                "top_k": 10,
                "top_p": 0.95,
                "truncate": None,
                "typical_p": 0.95,
                "watermark": True
            }
        }
        return request_body

    def _build_model_request_payload(self, prompt: str) -> FastChatRequestPayload:
        payload = FastChatRequestPayload(model=self._base_model_name, prompt=prompt, temperature=self._temperature)
        return payload


def _get_model_response_from_stream(response: Response) -> str:
    text_response = TEXT_RESPONSE_DEFAULT_VALUE
    for chunk in response.iter_lines(decode_unicode=False, delimiter=RESPONSE_CHUNK_DELIMITER):
        if chunk:
            data = json.loads(chunk.decode())
            chunk_text_response = data[0][RESPONSE_CHUNK_TEXT_FIELD]
            text_response = chunk_text_response

    return text_response

