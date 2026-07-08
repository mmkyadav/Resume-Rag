import os
import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenRouterService")

class OpenRouterClient:
    """
    Singleton wrapper for OpenRouter client execution, ensuring a single client
    instance is reused and configuration is centralized.
    """
    _instance: Optional["OpenRouterClient"] = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenRouterClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1", default_model: str = "meta-llama/llama-3.1-70b-instruct"):
        if self._initialized:
            return
            
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.default_model = default_model
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables or parameters.")
            
        # Re-use client. Configure with 30-second timeout
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0
        )
        self._initialized = True

    def get_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None, 
        temperature: float = 0.1, 
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> str:
        """
        Calls OpenRouter Chat Completions API with exponential backoff retries and timeout handling.
        """
        model_name = model or self.default_model
        
        # Reload API key if not initialized initially
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key:
                self.client.api_key = self.api_key
        
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            raise ValueError(
                "OPENROUTER_API_KEY is not set or contains the default placeholder. "
                "Please configure a valid OpenRouter API Key."
            )

        last_error = None
        current_delay = 1.0
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Sending request to OpenRouter using model '{model_name}' (Attempt {attempt}/{max_retries})...")
                
                extra_headers = {
                    "HTTP-Referer": "https://github.com/Antigravity/resume-rag-llm",
                    "X-Title": "Resume RAG LLM Recruiter Dashboard"
                }
                
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers
                )
                
                logger.info("OpenRouter request successful.")
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                logger.error(f"Error during OpenRouter request (Attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    sleep_time = current_delay
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor
                else:
                    logger.error("Max retries exceeded for OpenRouter query.")
                    
        raise RuntimeError(f"Failed to query OpenRouter after {max_retries} attempts. Last error: {last_error}")
