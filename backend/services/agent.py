"""
AI Agent Service - Using Google Gemini directly for AI responses.
Phase 13: Direct Gemini Integration
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from google.genai import types
from google.genai import Client

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Use just the model name without the provider prefix
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").replace("gemini/", "")

# Import MCP services
from services.sanity_mcp import (
    search_products as sanity_search_products,
    get_product_details as sanity_get_product_details,
    get_product_by_slug_tool as sanity_get_product_by_slug,
    get_products_by_category_tool as sanity_get_products_by_category,
    search_products_filtered as sanity_search_products_filtered,
    get_categories as sanity_get_categories,
    get_featured as sanity_get_featured
)

from services.database_mcp import (
    check_inventory as db_check_inventory,
    get_chat_history_tool as db_get_chat_history
)


# Agent Instructions
SYSTEM_PROMPT = """You are Sony Interior's virtual furniture consultant. Your role is to help customers find the perfect furniture for their needs.

## About Sony Interior
Sony Interior is a premium furniture store offering high-quality, modern furniture pieces including:
- Sofas and sofasets
- Chairs (dining, office, accent)
- Tables (dining, coffee, side)
- Beds and bedroom furniture
- Lighting fixtures
- Storage solutions

## Guidelines
1. Always be friendly, helpful, and professional
2. Provide detailed product information including prices, dimensions, materials, and availability
3. Help customers understand their options and make informed decisions
4. If a customer asks about something unrelated to furniture or Sony Interior, politely redirect them

## Response Format
- Always format product information clearly with name, price, and key features
- Include relevant details like dimensions and materials when available
- Mention if items are in stock or on backorder
- Suggest related products when appropriate
- Use bullet points for multiple options
- Be concise but informative

IMPORTANT: You have access to search product information from Sanity CMS. When customers ask about products, use your knowledge about furniture to provide helpful responses. If you need specific product details, suggest they browse the website or ask about specific items."""


# Global client
_gemini_client: Optional[Client] = None


async def get_gemini_client() -> Client:
    """Get or create Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = Client(api_key=GEMINI_API_KEY)
    return _gemini_client


async def run_agent_query(
    user_message: str,
    session_id: Optional[str] = None,
    page_context: Optional[str] = None,
    selected_text: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Run the agent with a user query using Gemini API."""
    try:
        client = await get_gemini_client()

        # Build context
        context_parts = []
        if page_context:
            context_parts.append(f"User is currently on page: {page_context}")
        if selected_text:
            context_parts.append(f"User selected this text: {selected_text}")

        context_info = "\n".join(context_parts)

        # Build chat history for context
        history_messages = []
        if chat_history and len(chat_history) > 0:
            for msg in chat_history[-5:]:
                role = "user" if msg.get("role") == "user" else "model"
                history_messages.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg.get("content", ""))]
                    )
                )

        # Initial message with context
        user_content = user_message
        if context_info:
            user_content = f"{context_info}\n\nUser message: {user_message}"

        contents = history_messages + [
            types.Content(
                role="user",
                parts=[types.Part(text=user_content)]
            )
        ]

        # Make the API call with system instruction in config
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
            )
        )

        # Extract response
        final_response = ""
        if response.candidates and response.candidates[0].content.parts:
            final_response = response.candidates[0].content.parts[0].text

        return {
            "response": final_response,
            "session_id": session_id,
            "success": True,
            "error": None
        }

    except Exception as e:
        print(f"Agent error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "session_id": session_id,
            "success": False,
            "error": str(e)
        }


# Standalone test function
async def test_agent():
    """Test the agent with sample queries."""
    print("Testing Sony Interior Agent...")

    test_queries = [
        "What featured products do you have?",
        "Show me some sofas",
        "What's the price of your best selling sofa?"
    ]

    for query in test_queries:
        print(f"\n\nQuery: {query}")
        result = await run_agent_query(query)
        print(f"Response: {result['response'][:500]}...")


if __name__ == "__main__":
    asyncio.run(test_agent())
