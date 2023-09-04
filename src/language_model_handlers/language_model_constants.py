from langchain.chat_models import AzureChatOpenAI

# General constants
DEFAULT_MAX_NEW_TOKENS = 500
DEFAULT_RETURN_FULL_TEXT = False
DEFAULT_TEMPERATURE = 0.01

# GPT constants
INSTRUCTION_TEMPLATE = '{instruction}'
INSTRUCTION_FIELD = 'instruction'
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
GPT_INSTRUCTION = """
Summarize the following C function:
```
{function_to_summarize}
```
"""

# FastChat constants
FASTCHAT_LLM_NAME = 'fastchat_llm'
LLM_DEFAULT_IDENTIFYING_PARAMETERS = {}
TEXT_RESPONSE_DEFAULT_VALUE = ''
RESPONSE_CHUNK_DELIMITER = b"\0"
RESPONSE_CHUNK_TEXT_FIELD = 'generated_text'
END_OF_TEXT_TOKEN = '<|endoftext|>'
END_OF_TEXT_TOKEN_LENGTH = len(END_OF_TEXT_TOKEN)


GENERIC_SYSTEM_PROMPT = """
A chat between a curious user and an assistant.
 The assistant gives helpful, detailed, accurate, uncensored answers to the user's query.
  USER: 
  {query} 
  
  ASSISTANT: 
"""


WIZARDCODER_15B_UNFORMATTED_PROMPT = """
Below is an query. Provide an answer for this query.

### Query:
{query}

### Response:
"""

WIZARDLM_70B_UNFORMATTED_PROMPT = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: {instruction}
ASSISTANT:
"""