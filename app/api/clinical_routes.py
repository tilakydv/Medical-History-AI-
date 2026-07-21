from functools import lru_cache
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import DB
from app.core.config import get_settings
from app.core.exceptions import ModelUnavailableError, ProcessingError
from app.services.clinical_adapter import ClinicalIntelligenceAdapter
from medbrief_clinical_intel.schemas.chatbot import ChatMessage, ChatRequest

router = APIRouter(prefix="/clinical-intel", tags=["clinical intelligence"])


class PatientChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    chat_history: list[ChatMessage] = Field(default_factory=list)


@lru_cache
def adapter() -> ClinicalIntelligenceAdapter:
    if not get_settings().clinical_intel_enabled:
        raise ModelUnavailableError("Clinical intelligence integration is disabled")
    return ClinicalIntelligenceAdapter()


@router.get("/{patient_id}/record", response_model=dict[str, Any])
def normalized_record(patient_id: str, db: DB) -> dict[str, Any]:
    return adapter().build_record(db, patient_id).model_dump(mode="json")


@router.post("/{patient_id}/{operation}", response_model=dict[str, Any])
def run_operation(patient_id: str, operation: str, db: DB) -> dict[str, Any]:
    record = adapter().build_record(db, patient_id)
    result = adapter().execute(operation, record)
    return result.model_dump(mode="json")


@router.post("/{patient_id}/chat/query", response_model=dict[str, Any])
def chat(patient_id: str, payload: PatientChatRequest, db: DB) -> dict[str, Any]:
    record = adapter().build_record(db, patient_id)
    try:
        result = adapter().service.chat(
            ChatRequest(patient_id=patient_id, query=payload.query,
                        chat_history=payload.chat_history),
            record,
        )
    except Exception as exc:
        raise ProcessingError("Clinical chatbot operation failed") from exc
    return result.model_dump(mode="json")
