"""
Robust JSON output parsing and validation utilities for Qwen2.5 responses.
"""

import json
import re
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JSONOutputParser:
    """Parser to clean LLM markdown code blocks and parse structured JSON into Pydantic models."""

    @staticmethod
    def extract_json_str(text: str) -> str:
        """Extract JSON substring from raw model output text."""
        text = text.strip()
        # Look for markdown ```json ... ``` codeblock
        json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if json_block_match:
            return json_block_match.group(1).strip()
        
        # Look for curly braces { ... } or brackets [ ... ]
        object_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if object_match:
            return object_match.group(1).strip()
            
        return text

    @classmethod
    def parse_json(cls, text: str) -> Any:
        """Parse raw LLM output into python dict/list."""
        cleaned = cls.extract_json_str(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Clean common trailing commas or single quote issues if any
            cleaned_fix = re.sub(r",\s*([\}\]])", r"\1", cleaned)
            return json.loads(cleaned_fix)

    @classmethod
    def parse_to_model(cls, text: str, model_cls: Type[T]) -> Optional[T]:
        """Parse raw LLM output into specified Pydantic model instance."""
        try:
            parsed_data = cls.parse_json(text)
            if isinstance(parsed_data, dict):
                return model_cls.model_validate(parsed_data)
            elif isinstance(parsed_data, list) and hasattr(model_cls, "model_validate"):
                return model_cls.model_validate({"items": parsed_data})
        except Exception as e:
            # Could not directly validate model
            return None
        return None


