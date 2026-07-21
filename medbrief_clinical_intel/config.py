"""
Configuration settings for MedBrief AI Clinical Intelligence Module
"""

import os
from pydantic import BaseModel, Field

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

default_config = ClinicalIntelConfig()


