import openai
import logging
from typing import Dict, Any, Optional, List
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # Get current provider configuration
        self.config = settings.get_current_provider_config()
        self.provider = self.config["provider"]
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
        self.model_name = self.config.get("model_name", "gpt-4o")  # Default fallback "gpt-4o"
        
        # Validate configuration
        validation = settings.validate_configuration()
        if not validation["valid"]:
            raise ValueError(f"Configuration errors: {validation['errors']}")
        
        # Initialize OpenAI client
        if self.provider in ["tiger", "lmstudio"]:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        
        logger.info(f"✅ {self.provider.upper()} client initialized")
        logger.info(f"   Default Model: {self.model_name}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None  # <-- NEW argument for per-call override
    ) -> Optional[str]:
        """Make chat completion request"""
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = self.client.chat.completions.create(
                model=model or self.model_name or "gpt-4o",  # <-- per-call override or default
                messages=messages,
                temperature=temperature if temperature is not None else settings.TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ {self.provider.upper()} API error: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test connection"""
        try:
            result = await self.simple_completion("Hello", "You are a test assistant.")
            return bool(result)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def simple_completion(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        model: Optional[str] = None  # <-- add model override for simple calls too
    ) -> Optional[str]:
        """Simple completion"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, system_prompt=system_prompt, model=model)

# Global instance
llm_client = LLMClient()