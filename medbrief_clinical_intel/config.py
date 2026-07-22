"""
Configuration settings for MedBrief AI Clinical Intelligence Module
"""

import os
from typing import Optional
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



class ClinicalIntelConfig(BaseModel):
    """Configuration for LLM and Clinical Intelligence Module."""
    model_name_or_path: str = Field(
        default=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct"),
        description="HuggingFace model ID or local path to Qwen2.5-14B-Instruct"
    )
    device: str = Field(
        default=os.getenv("MEDBRIEF_DEVICE", "auto"),
        description="Device to load model on ('auto', 'cuda', 'cpu')"
    )
    load_in_4bit: bool = Field(
        default=os.getenv("LOAD_IN_4BIT", "false").lower() == "true",
        description="Whether to use bitsandbytes 4-bit quantization"
    )
    load_in_8bit: bool = Field(
        default=os.getenv("LOAD_IN_8BIT", "false").lower() == "true",
        description="Whether to use bitsandbytes 8-bit quantization"
    )
    max_new_tokens: int = Field(
        default=2048,
        description="Maximum new tokens generated per LLM request"
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for deterministic clinical reasoning"
    )
    use_mock_llm: bool = Field(
        default=os.getenv("USE_MOCK_LLM", "true").lower() == "true",
        description="If True, uses rule-based deterministic mock execution when LLM is unavailable"
    )
    llm_provider: str = Field(
        default=os.getenv("LLM_PROVIDER", "mock"),
        description="LLM provider: 'mock', 'huggingface', 'gemini', 'openai', 'ollama'"
    )
    chatbot_only_llm: bool = Field(
        default=os.getenv("CHATBOT_ONLY_LLM", "true").lower() == "true",
        description="If True, LLM generates content ONLY for chatbot queries, keeping other modules deterministic"
    )
    gemini_api_key: Optional[str] = Field(
        default=os.getenv("GEMINI_API_KEY"),
        description="Google Gemini API key"
    )
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )
    openai_api_base: str = Field(
        default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        description="OpenAI API base URL"
    )
    openai_model: str = Field(
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        description="OpenAI model name"
    )
    ollama_api_url: str = Field(
        default=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        description="Ollama model name"
    )

default_config = ClinicalIntelConfig()



