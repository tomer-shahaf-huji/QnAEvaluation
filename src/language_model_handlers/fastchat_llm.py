import json
from typing import Any, Mapping

import requests
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from requests.models import Response

from src.language_model_handlers.language_model_constants import DEFAULT_TEMPERATURE, FASTCHAT_LLM_NAME, \
    LLM_DEFAULT_IDENTIFYING_PARAMETERS, TEXT_RESPONSE_DEFAULT_VALUE, RESPONSE_CHUNK_DELIMITER, \
    RESPONSE_CHUNK_TEXT_FIELD, END_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN_LENGTH, DEFAULT_MAX_NEW_TOKENS, \
    DEFAULT_RETURN_FULL_TEXT, WIZARDCODER_15B_UNFORMATTED_PROMPT


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

    def _call(self, instruction: str, **kwargs: Any) -> str:
        request_headers = self._build_request_headers()
        request_body = self._build_request_body(instruction)
        response = requests.post(self._fastchat_url, headers=request_headers, json=request_body)
        response.raise_for_status()  # ToDo: Is this necessary?
        response_text = _get_model_response_from_stream(response)
        return response_text

    def _build_request_headers(self) -> Mapping[str, str]:
        request_headers = {'User-Agent': 'FastChat Client', 'Host': self._model_host}
        return request_headers

    def _build_request_body(self, instruction: str) -> dict:
        prompt = WIZARDCODER_15B_UNFORMATTED_PROMPT.format(query=instruction)
        request_body = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "return_full_text": DEFAULT_RETURN_FULL_TEXT,
                "temperature": self._temperature
            }
        }
        return request_body


def _get_model_response_from_stream(response: Response) -> str:
    text_response = TEXT_RESPONSE_DEFAULT_VALUE
    for chunk in response.iter_lines(decode_unicode=False, delimiter=RESPONSE_CHUNK_DELIMITER):
        if chunk:
            data = json.loads(chunk.decode())
            chunk_text_response = data[0][RESPONSE_CHUNK_TEXT_FIELD]
            text_response = chunk_text_response
    if text_response.endswith(END_OF_TEXT_TOKEN):
        text_response = text_response[:-END_OF_TEXT_TOKEN_LENGTH]
    return text_response