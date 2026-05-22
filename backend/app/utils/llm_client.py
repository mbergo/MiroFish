"""
LLM客户端封装
支持 OpenAI 兼容格式和 Anthropic 原生 SDK
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config

try:
    import anthropic as anthropic_sdk
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _is_anthropic_provider(base_url: Optional[str], model: Optional[str]) -> bool:
    if base_url and "anthropic.com" in base_url:
        return True
    if model and model.startswith("claude-"):
        return True
    return False


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self._use_anthropic = _is_anthropic_provider(self.base_url, self.model)

        if self._use_anthropic:
            if not HAS_ANTHROPIC:
                raise ImportError("使用 Anthropic 模型需要安装: pip install anthropic")
            self.client = anthropic_sdk.Anthropic(api_key=self.api_key)
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        if self._use_anthropic:
            return self._chat_anthropic(messages, temperature, max_tokens)
        return self._chat_openai(messages, temperature, max_tokens, response_format)

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict]
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Anthropic separates system prompt from conversation turns
        system: Optional[str] = None
        turns = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                turns.append({"role": m["role"], "content": m["content"]})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        if self._use_anthropic:
            # Anthropic has no response_format param — use prefill trick to force JSON
            turns = list(messages)
            turns.append({
                "role": "user",
                "content": "Respond with valid JSON only. No markdown, no code blocks, no explanation."
            })
            # Pre-fill assistant turn with "{" — Claude continues from here
            turns.append({"role": "assistant", "content": "{"})
            raw = self._chat_anthropic(turns, temperature, max_tokens)
            # Restore the pre-filled "{" that the API strips from the response
            response = "{" + raw
        else:
            response = self._chat_openai(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned}")
