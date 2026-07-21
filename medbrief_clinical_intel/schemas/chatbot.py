"""
Schema for AI Clinical RAG Chatbot (Module 3).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Previous chat message in conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """User query input to RAG chatbot."""
    patient_id: str
    query: str = Field(..., description="Doctor or user clinical question")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Optional conversation context")


class Citation(BaseModel):
    """Document citation linking chatbot claims back to patient records."""
    document_id: Optional[str] = Field(default=None, description="Source report or document ID")
    excerpt: Optional[str] = Field(default=None, description="Exact or summarized text excerpt from source")


class ChatResponse(BaseModel):
    """Grounded chatbot response strictly based on patient records."""
    patient_id: str
    query: str
    answer: str = Field(..., description="Clinical response strictly derived from patient records")
    is_grounded: bool = Field(default=True, description="True if response is fully supported by uploaded files")
    not_found_in_records: bool = Field(default=False, description="True if query cannot be answered from uploaded records")
    citations: List[Citation] = Field(default_factory=list, description="Supporting document references")


