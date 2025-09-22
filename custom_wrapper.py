import requests
from typing import List, Optional
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    ChatGeneration,
    HumanMessage,
)
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field


class OpenRouterChat(BaseChatModel):
    api_key: str = Field(...)
    model: str = "mistralai/mistral-7b-instruct:free"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

    def _format_message(self, message: BaseMessage) -> dict:
        role = "user"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        return {"role": role, "content": message.content}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourdomain.com",
            "X-Title": "LangChainOpenRouterWrapper"
        }

        payload = {
            "model": self.model,
            "messages": [self._format_message(m) for m in messages],
            "temperature": self.temperature
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")

        content = response.json()["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )
