from pydantic import BaseModel


class FastChatRequestPayload(BaseModel):
    model: str
    prompt: str
    temperature: float