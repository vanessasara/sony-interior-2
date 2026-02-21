"""
Chat router - Handles chat API endpoints.
Phase 14: FastAPI Chat Endpoint
"""

import os
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from models.chat_models import (
    ChatRequest,
    ChatResponse,
    QuickQuestionsRequest,
    QuickQuestionsResponse,
    SessionCreateRequest,
    SessionResponse,
    ChatHistoryResponse,
    Message
)
from services.agent import run_agent_query
from services.database_mcp import get_db_server, create_session as db_create_session

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["chat"])

# In-memory session storage (for development - use database in production)
# Format: {session_id: {"created_at": datetime, "messages": [], "current_page": str}}
sessions: Dict[str, Dict[str, Any]] = {}

# Quick questions by page type
QUICK_QUESTIONS = {
    "product": [
        "What are the dimensions of this item?",
        "Is this currently in stock?",
        "What materials is this made from?",
        "Show me similar products",
        "What's the price and any current deals?"
    ],
    "products": [
        "What are your bestselling items?",
        "Show me sofas under $1000",
        "What new products arrived recently?",
        "Help me find a dining table",
        "What furniture categories do you offer?"
    ],
    "home": [
        "What makes Sony Interior unique?",
        "Tell me about your featured collection",
        "How can I visit your showroom?",
        "What's your return policy?",
        "What are your bestsellers?"
    ],
    "about": [
        "Where are you located?",
        "Tell me about your craftsmanship",
        "What's your sustainability approach?",
        "How long have you been in business?",
        "Do you offer interior design services?"
    ],
    "contact": [
        "What are your store hours?",
        "How can I schedule a showroom visit?",
        "Do you offer delivery services?",
        "What is your return policy?",
        "How can I contact customer support?"
    ],
    "default": [
        "Help me find the perfect furniture",
        "What are your featured products?",
        "Tell me about your company",
        "What categories do you offer?",
        "How can I contact you?"
    ]
}


def generate_session_id() -> str:
    """Generate a new session ID."""
    return str(uuid.uuid4())


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """
    Main chat endpoint - receives user message and returns agent response.

    Args:
        request: Chat request with message and optional context
        req: FastAPI request object

    Returns:
        ChatResponse with agent's reply
    """
    try:
        # Get or create session ID
        session_id = request.session_id or generate_session_id()

        # Get chat history for context
        chat_history = sessions.get(session_id, {}).get("messages", [])

        # Build page context
        page_context = request.page_context or req.headers.get("X-Page-Context", "/")

        # Run agent query
        logger.info(f"Processing chat message for session {session_id}: {request.message[:50]}...")

        result = await run_agent_query(
            user_message=request.message,
            session_id=session_id,
            page_context=page_context,
            selected_text=request.selected_text,
            chat_history=chat_history
        )

        # Store messages in session
        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now(),
                "messages": [],
                "current_page": page_context
            }

        # Add user message
        sessions[session_id]["messages"].append({
            "role": "user",
            "content": request.message,
            "created_at": datetime.now().isoformat()
        })

        # Add assistant response
        sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": result.get("response", ""),
            "created_at": datetime.now().isoformat()
        })

        # Try to save to database (if available)
        try:
            db_server = await get_db_server()
            await db_server.save_chat_message(
                session_id=session_id,
                role="user",
                content=request.message,
                page_context=page_context
            )
            await db_server.save_chat_message(
                session_id=session_id,
                role="assistant",
                content=result.get("response", "")
            )
        except Exception as e:
            logger.warning(f"Failed to save to database: {e}")

        # Update session page
        sessions[session_id]["current_page"] = page_context

        return ChatResponse(
            response=result.get("response", "I apologize, but I couldn't generate a response."),
            session_id=session_id,
            success=result.get("success", True),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """
    Streaming chat endpoint - streams agent response token by token.

    Note: This is a placeholder. Full streaming implementation requires
    modifying the agent to yield chunks instead of waiting for complete response.
    """
    # For now, return a simple streaming response
    # Full streaming would require modifying the agent to support async generation

    session_id = request.session_id or generate_session_id()

    async def generate():
        # First, yield the session ID
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        # Then process the message
        try:
            page_context = request.page_context or "/"
            result = await run_agent_query(
                user_message=request.message,
                session_id=session_id,
                page_context=page_context,
                selected_text=request.selected_text
            )

            response_text = result.get("response", "")

            # Stream the response in chunks
            chunk_size = 20
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/quick-questions", response_model=QuickQuestionsResponse)
async def get_quick_questions(page_type: str = "default", product_id: Optional[str] = None):
    """
    Get quick questions based on current page.

    Args:
        page_type: Type of page (home, product, products, about, contact)
        product_id: Optional product ID for product-specific questions

    Returns:
        List of quick questions
    """
    questions = QUICK_QUESTIONS.get(page_type, QUICK_QUESTIONS["default"])

    # If on a product page, add product-specific questions
    if page_type == "product" and product_id:
        product_questions = [
            f"What are the dimensions of this product?",
            f"Is this product in stock?",
            f"What materials is this made from?"
        ]
        questions = product_questions + questions[:2]

    return QuickQuestionsResponse(questions=questions)


@router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new chat session.

    Args:
        request: Session creation request

    Returns:
        Session ID
    """
    session_id = generate_session_id()

    # Store in memory
    sessions[session_id] = {
        "created_at": datetime.now(),
        "messages": [],
        "current_page": request.current_page or "/"
    }

    # Try to create in database
    try:
        db_server = await get_db_server()
        await db_server.create_chat_session(
            user_agent=request.user_agent,
            current_page=request.current_page,
            metadata=request.metadata
        )
    except Exception as e:
        logger.warning(f"Failed to create session in database: {e}")

    return SessionResponse(
        session_id=session_id,
        success=True,
        message="Session created successfully"
    )


@router.get("/session/{session_id}", response_model=ChatHistoryResponse)
async def get_session_history(session_id: str, limit: int = 50):
    """
    Get chat history for a session.

    Args:
        session_id: Session ID
        limit: Maximum messages to return

    Returns:
        Chat history
    """
    # Check in-memory storage first
    if session_id in sessions:
        messages = sessions[session_id].get("messages", [])[-limit:]
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[Message(**msg) for msg in messages],
            success=True
        )

    # Try database
    try:
        db_server = await get_db_server()
        history = await db_server.get_chat_history(session_id, limit)
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[Message(**msg) for msg in history],
            success=True
        )
    except Exception as e:
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[],
            success=False,
            error=str(e)
        )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session.

    Args:
        session_id: Session ID to delete

    Returns:
        Success message
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"success": True, "message": "Session deleted"}

    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health/chat")
async def chat_health_check():
    """
    Health check for chat service.
    """
    return {
        "status": "healthy",
        "service": "chat",
        "sessions_active": len(sessions),
        "model": os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
    }
