"""
Pydantic models for chat API requests and responses.
Phase 14: FastAPI Chat Endpoint
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Chat session identifier")
    page_context: Optional[str] = Field(None, description="Current page URL")
    selected_text: Optional[str] = Field(None, description="Text user highlighted on page")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent's reply")
    session_id: str = Field(..., description="Session identifier")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Products or content sources used")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if failed")


class QuickQuestionsRequest(BaseModel):
    """Request model for quick questions generation."""
    page_type: str = Field(..., description="Type of page (home, product, products, about, contact)")
    product_id: Optional[str] = Field(None, description="Product ID if on product page")


class QuickQuestionsResponse(BaseModel):
    """Response model for quick questions."""
    questions: List[str] = Field(..., description="List of generated questions")


class SessionCreateRequest(BaseModel):
    """Request model for creating a new session."""
    user_agent: Optional[str] = Field(None, description="User's browser info")
    current_page: Optional[str] = Field(None, description="Initial page URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional session metadata")


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    success: bool = True
    message: Optional[str] = None


class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    created_at: Optional[datetime] = Field(None, description="Message timestamp")
    token_usage: Optional[int] = Field(None, description="Token count for this message")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    session_id: str
    messages: List[Message]
    success: bool = True
    error: Optional[str] = None
