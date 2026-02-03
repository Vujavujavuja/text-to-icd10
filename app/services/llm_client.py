"""
LLM Client for OpenRouter API

Handles communication with OpenRouter for clinical text processing.
"""

import httpx
from typing import Dict, Any, Optional
from loguru import logger


class LLMClient:
    """Client for OpenRouter API using free models."""

    def __init__(self, api_key: str, model: str = "anthropic/claude-haiku-4.5"):
        """
        Initialize LLM client.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: qwen free model)
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self.timeout = 180.0  # 3 minutes for slower models like Kimi

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000
    ) -> str:
        """
        Call OpenRouter API for text generation.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text from LLM
        """
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            logger.info(f"Calling OpenRouter API with model: {self.model}, timeout: {self.timeout}s")
            logger.debug(f"Prompt length: {len(prompt)} chars")

            timeout_config = httpx.Timeout(self.timeout, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                logger.info("Sending POST request to OpenRouter...")
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/icd10-coding-assistant",
                        "X-Title": "ICD-10 Coding Assistant"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "provider": {
                            "allow_fallbacks": True,
                            "order": ["anthropic"],
                            "ignore": ["chutes"]
                        }
                    }
                )
                logger.info(f"Received response from OpenRouter: status={response.status_code}")
                response.raise_for_status()
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                logger.info(f"LLM response length: {len(response_text)} chars")
                return response_text

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"LLM API error: {e.response.status_code}")
        except httpx.TimeoutException as e:
            logger.error(f"OpenRouter API timeout after {self.timeout}s: {e}")
            raise Exception(f"LLM API timeout after {self.timeout}s")
        except httpx.ConnectError as e:
            logger.error(f"OpenRouter API connection error: {e}")
            raise Exception(f"LLM API connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in LLM generation: {type(e).__name__}: {e}", exc_info=True)
            raise

    def is_ready(self) -> bool:
        """Check if LLM client is configured."""
        return bool(self.api_key)
