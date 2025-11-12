"""
Anthropic Claude LLM Client Implementation
Provides an interface compatible with GeminiLLMClient so the rest of the app can
use Claude the same way as Gemini.

Notes:
- This module is defensive about Anthropic SDK shape; it will try to use the
  Messages API when available and fall back to a prompt/completions-style call.
- You must install an Anthropic Python package (commonly `anthropic`) and set
  `CLAUDE_API_KEY` (and optionally `CLAUDE_MODEL_NAME`) in the environment or
  `.env` for `from_environment()` to work.
"""

import os
import sys
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from cli import console
from pydantic import BaseModel


class ClaudeConfig(BaseModel):
    model_name: str = os.getenv('CLAUDE_MODEL_NAME', 'claude-2.1')
    claude_api_key: str = os.getenv('CLAUDE_API_KEY', '')


class ClaudeChatSessionWrapper:
    """Wrapper exposing send_message() and get_history() similar to Gemini."""

    def __init__(self, client, model_name: str, system_instruction: str = ''):
        self.client = client
        self.model_name = model_name
        self.system_instruction = system_instruction or ''
        # universal history format: list of {"role": "user|model|system", "parts": [{"text": "..."}]}
        self._history: List[Dict] = []

    def send_message(self, text: str) -> Any:
        # Append user message to universal history
        self._history.append({"role": "user", "parts": [{"text": text}]})

        try:
            # Build messages structure for Anthropic Messages API
            messages = []
            
            for entry in self._history:
                role = entry.get('role')
                text_part = entry.get('parts', [{}])[0].get('text', '')
                
                if role == 'user':
                    messages.append({"role": "user", "content": text_part})
                elif role in ('model', 'assistant'):
                    messages.append({"role": "assistant", "content": text_part})

            response = None

            # Messages API (preferred)
            if hasattr(self.client, 'messages') and hasattr(self.client.messages, 'create'):
                try:
                    # Prepare API call parameters
                    api_params = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": 4096  # Required parameter
                    }
                    
                    # Add system instruction as separate parameter if present
                    if self.system_instruction:
                        api_params["system"] = self.system_instruction
                    
                    response = self.client.messages.create(**api_params)
                    console.print_info("Użyto Anthropic Messages API")
                    
                    # Extract text from Messages API response
                    if hasattr(response, 'content') and response.content:
                        # response.content is a list of content blocks
                        text_out = ''
                        for block in response.content:
                            if hasattr(block, 'text'):
                                text_out += block.text
                        
                        # Append model response into universal history
                        self._history.append({"role": "model", "parts": [{"text": text_out}]})
                        return SimpleResponse(text_out)
                        
                except Exception as e:
                    console.print_error(f"Messages API error: {e}")
                    raise

            # If we get here, Messages API failed
            raise RuntimeError("Messages API not available or failed")

        except Exception as e:
            console.print_error(f"Błąd podczas wywołania Claude: {e}")
            err_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi Claude."
            self._history.append({"role": "model", "parts": [{"text": err_text}]})
            return SimpleResponse(err_text)

    def get_history(self) -> List[Dict]:
        return self._history


class SimpleResponse:
    def __init__(self, text: str):
        self.text = text


class ClaudeLLMClient:
    """Client class compatible with ChatSession expectations (from_environment, create_chat_session, etc.)"""

    def __init__(self, model_name: str, claude_api_key: str):
        if not claude_api_key:
            raise ValueError("Claude API key cannot be empty")
        self.model_name = model_name
        self.api_key = claude_api_key
        self._client = self._initialize_client()

    @staticmethod
    def preparing_for_use_message() -> str:
        return " Przygotowywanie klienta Claude..."

    @classmethod
    def from_environment(cls) -> 'ClaudeLLMClient':
        load_dotenv()
        config = ClaudeConfig()
        return cls(model_name=config.model_name, claude_api_key=config.claude_api_key)

    def _initialize_client(self):
        try:
            import anthropic
            # instantiate client in common form
            # different anthopic libs expect different ctor signatures; try common ones
            try:
                client = anthropic.Client(api_key=self.api_key)
            except Exception:
                try:
                    client = anthropic.Anthropic(api_key=self.api_key)
                except Exception as e:
                    raise
            return client
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta Claude: {e}")
            sys.exit(1)

    def create_chat_session(self, system_instruction: str, history: Optional[List[Dict]] = None, thinking_budget: int = 0) -> ClaudeChatSessionWrapper:
        wrapper = ClaudeChatSessionWrapper(self._client, self.model_name, system_instruction=system_instruction)
        if history:
            for entry in history:
                if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
                    wrapper._history.append(entry)
        return wrapper

    def count_history_tokens(self, history: List[Dict]) -> int:
        # Best-effort estimate
        if not history:
            return 0
        total_chars = sum(len(h.get('parts', [{}])[0].get('text', '')) for h in history)
        return total_chars // 4

    def get_model_name(self) -> str:
        return self.model_name

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key)

    def ready_for_use_message(self) -> str:
        masked = self.api_key[:4] + '...' + self.api_key[-4:] if len(self.api_key) > 8 else '****'
        return f"\u2705 Klient Claude gotowy (Model: {self.model_name}, Key: {masked})"

    @property
    def client(self):
        return self._client
