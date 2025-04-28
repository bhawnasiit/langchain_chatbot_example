from typing import Any, List
from pydantic import Field
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import HumanMessage, AIMessage
import requests
import os

class ChatDeepSeek(SimpleChatModel):
    model: str = Field(default="deepseek-chat")
    openai_api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    openai_api_base: str = Field(default="https://api.deepseek.com")
    temperature: float = 0.7
    max_tokens: int = 512

    def _call(self, messages: List[Any], **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": msg.type, "content": msg.content} for msg in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{self.openai_api_base}/v1/chat/completions", json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
