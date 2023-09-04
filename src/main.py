from langchain import PromptTemplate
from langchain.chains import LLMChain

from src.language_model_handlers.fastchat_llm import FastChatLLM
from src.language_model_handlers.language_model_constants import WIZARDCODER_15B_UNFORMATTED_PROMPT, \
    VICUNIA_SYSTEM_PROMPT

QUERY = 'What is the the capitol of Spain?'

if __name__ == '__main__':
    wizardcoder_15b = FastChatLLM(
        model_host='model.llm.com',
        fastchat_url='http://20.246.192.153',
        base_model_name='WizardLM/WizardCoder-15B-V1.0'
    )

    PROMPT = PromptTemplate(
        input_variables=["query"],
        template=VICUNIA_SYSTEM_PROMPT
    )

    wizard_chain = LLMChain(llm=wizardcoder_15b, prompt=PROMPT)
    chain_response = wizard_chain.run(query=QUERY)
    print(chain_response)




