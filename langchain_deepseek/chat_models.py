from typing import Any, List
from pydantic import Field
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
            "messages": [{"role": self._convert_role(msg), "content": msg.content} for msg in messages],
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

    def _convert_role(self, message: Any) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "user"  # Default fallback for any unknown message

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
