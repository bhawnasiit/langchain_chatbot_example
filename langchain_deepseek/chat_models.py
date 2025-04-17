from langchain.chat_models.base import SimpleChatModel
from langchain.schema import AIMessage, HumanMessage
import requests
import os

class ChatDeepSeek(SimpleChatModel):
    def __init__(self, model="deepseek-chat", openai_api_key=None, openai_api_base=None, temperature=0.7, max_tokens=512):
        self.model = model
        self.api_key = openai_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = openai_api_base or "https://api.deepseek.com"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": [{"role": msg.type, "content": msg.content} for msg in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(f"{self.api_base}/v1/chat/completions", json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _identifying_params(self):
        return {"model": self.model}
