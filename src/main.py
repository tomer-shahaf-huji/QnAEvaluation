from src.llms_api.openai_client import OpenAIClient

if __name__ == '__main__':
    client = OpenAIClient()
    prompt = "What is the capital of Spain?"
    response = client.generate_response(prompt)
    print(response)
