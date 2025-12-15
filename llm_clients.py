"""
LLM Client integrations with async support and error handling
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.0, 
                      max_tokens: int = 2000) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured response following schema"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client with structured output support"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name, config.API_KEYS["openai"])
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)
        self._new_param_keywords = (
            "gpt-4.1",
            "gpt-5",
            "o1",
            "o2",
            "o3",
            "o4",
            "mini",
        )
        self._fixed_temperature_keywords = self._new_param_keywords

    def _completion_kwargs(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not any(keyword in self.model_name for keyword in self._fixed_temperature_keywords):
            kwargs["temperature"] = temperature
        
        if any(keyword in self.model_name for keyword in self._new_param_keywords):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        return kwargs
    
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 2000) -> str:
        """Generate text response"""
        try:
            response = await self.client.chat.completions.create(
                **self._completion_kwargs(prompt, temperature, max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured JSON response"""
        try:
            from pydantic import BaseModel, create_model
            
            # Create Pydantic model from schema
            fields = {k: (v.get("type", str), ...) for k, v in response_schema.get("properties", {}).items()}
            ResponseModel = create_model("ResponseModel", **fields)
            
            completion_kwargs = self._completion_kwargs(prompt, temperature, max_tokens=2000)
            response = await self.client.beta.chat.completions.parse(
                response_format=ResponseModel,
                **completion_kwargs
            )
            return response.choices[0].message.parsed.dict()
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to regular: {e}")
            # Fallback to regular generation
            text = await self.generate(prompt, temperature)
            return {"raw_text": text}


class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(model_name, config.API_KEYS["anthropic"])
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            default_headers={"anthropic-version": "2023-06-01"}
        )
    
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 2000) -> str:
        """Generate text response"""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured response (uses prompt engineering)"""
        import json
        structured_prompt = f"{prompt}\n\nRespond with a JSON object following this schema:\n{json.dumps(response_schema, indent=2)}"
        text = await self.generate(structured_prompt, temperature)
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_text": text}


class GoogleClient(LLMClient):
    """Google Gemini API client"""
    
    def __init__(self, model_name: str = "gemini-1.5-pro-latest"):
        super().__init__(model_name, config.API_KEYS["google"])
        if not self.api_key:
            raise ValueError("Google API key is not configured")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 8000) -> str:
        """Generate text response"""
        try:
            endpoint = f"{self.base_url}/models/{self.model_name}:generateContent"
            params = {"key": self.api_key}
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, params=params, json=payload) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise RuntimeError(f"Google API error {response.status}: {error_text}")
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        # Log the block reason for debugging
                        prompt_feedback = data.get("promptFeedback", {})
                        block_reason = prompt_feedback.get("blockReason", "unknown")
                        logger.warning(f"Google API returned no candidates. Block reason: {block_reason}, promptFeedback: {prompt_feedback}")
                        return ""
                    
                    # Log finish reason for debugging truncation issues
                    finish_reason = candidates[0].get("finishReason", "unknown")
                    if finish_reason == "MAX_TOKENS":
                        logger.warning(f"Google API response truncated due to MAX_TOKENS limit")
                    
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                    return text.strip()
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
    
    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured response"""
        import json
        structured_prompt = f"{prompt}\n\nRespond with a JSON object following this schema:\n{json.dumps(response_schema, indent=2)}"
        text = await self.generate(structured_prompt, temperature)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_text": text}


class GrokClient(LLMClient):
    """xAI Grok API client (OpenAI-compatible)"""

    def __init__(self, model_name: str = "grok-beta"):
        super().__init__(model_name, config.API_KEYS["grok"])
        if not self.api_key:
            raise ValueError("Grok API key is not configured")
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 2000) -> str:
        """Generate text response using xAI Grok."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            raise

    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured response via prompt engineering."""
        import json
        structured_prompt = f"{prompt}\n\nRespond with a JSON object following this schema:\n{json.dumps(response_schema, indent=2)}"
        text = await self.generate(structured_prompt, temperature)

        try:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return {"raw_text": text}


class DeepSeekClient(LLMClient):
    """DeepSeek API client (OpenAI-compatible)"""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__(model_name, config.API_KEYS["deepseek"])
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
    
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 2000) -> str:
        """Generate text response"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured response"""
        import json
        structured_prompt = f"{prompt}\n\nRespond with a JSON object following this schema:\n{json.dumps(response_schema, indent=2)}"
        text = await self.generate(structured_prompt, temperature)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_text": text}


class OllamaClient(LLMClient):
    """Ollama local model client"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        super().__init__(model_name, None)
        self.base_url = base_url
    
    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=config.RETRY_DELAY))
    async def generate(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 2000) -> str:
        """Generate text response"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    }
                ) as response:
                    result = await response.json()
                    return result["response"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def generate_structured(self, prompt: str, response_schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Generate structured response"""
        import json
        structured_prompt = f"{prompt}\n\nRespond with a JSON object following this schema:\n{json.dumps(response_schema, indent=2)}"
        text = await self.generate(structured_prompt, temperature)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_text": text}


def get_client(provider: str, model_name: str) -> LLMClient:
    """Factory function to get appropriate client"""
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "grok": GrokClient,
        "deepseek": DeepSeekClient,
        "ollama": OllamaClient,
    }
    
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}")
    
    return clients[provider](model_name)


async def batch_generate(clients: List[LLMClient], prompts: List[str],
                        temperature: float = 0.0) -> List[str]:
    """Generate responses from multiple clients in parallel"""
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    
    async def generate_with_limit(client, prompt):
        async with semaphore:
            return await client.generate(prompt, temperature)
    
    tasks = [generate_with_limit(client, prompt) 
             for client, prompt in zip(clients, prompts)]
    return await asyncio.gather(*tasks, return_exceptions=True)
