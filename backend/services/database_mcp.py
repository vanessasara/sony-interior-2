"""
Database MCP Server - Provides tools for querying Neon database.
Phase 12: MCP Server Development
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional, List
import asyncpg

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "")


class DatabaseMCPServer:
    """MCP Server for Neon database queries."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize database connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10
            )
            print("Database connection pool initialized")

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def _execute(self, query: str, *args) -> Any:
        """Execute a query and return results."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            if query.strip().upper().startswith("SELECT"):
                return await conn.fetch(query, *args)
            else:
                return await conn.execute(query, *args)

    async def check_product_stock(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Check stock status for a product.

        Args:
            product_id: Product ID to check

        Returns:
            Stock information or None
        """
        # First check product_embeddings table
        result = await self._execute(
            """
            SELECT source_id, metadata
            FROM document_embeddings
            WHERE source_type = 'product' AND source_id = $1
            LIMIT 1
            """,
            product_id
        )

        if result:
            return {
                "product_id": product_id,
                "stock_status": result[0].get("metadata", {}).get("stock_status", "unknown"),
                "available": True
            }

        return {
            "product_id": product_id,
            "stock_status": "unknown",
            "available": False,
            "message": "Product not found in database"
        }

    async def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get chat message history for a session.

        Args:
            session_id: Chat session ID
            limit: Maximum messages to return

        Returns:
            List of chat messages
        """
        results = await self._execute(
            """
            SELECT message_id, session_id, role, content, created_at, token_usage, page_context
            FROM chat_messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
            """,
            session_id,
            limit
        )

        return [
            {
                "message_id": str(r["message_id"]),
                "session_id": str(r["session_id"]),
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "token_usage": r["token_usage"],
                "page_context": r["page_context"]
            }
            for r in results
        ]

    async def search_similar_content(
        self,
        query_text: str,
        query_embedding: List[float],
        source_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Note: This requires pgvector extension to be installed.
        Without pgvector, returns a placeholder message.

        Args:
            query_text: Original query text
            query_embedding: Query embedding vector
            source_type: Optional filter by source type
            limit: Maximum results

        Returns:
            List of similar content chunks
        """
        # pgvector not available - return placeholder
        return [
            {
                "source_type": "info",
                "source_id": "system",
                "content": "Vector search is not available. The database needs pgvector extension enabled.",
                "similarity": 0.0,
                "metadata": {}
            }
        ]

    async def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_usage: Optional[int] = None,
        page_context: Optional[str] = None
    ) -> str:
        """
        Save a chat message to the database.

        Args:
            session_id: Chat session ID
            role: Message role (user/assistant)
            content: Message content
            token_usage: Optional token count
            page_context: Optional page context

        Returns:
            Message ID
        """
        result = await self._execute(
            """
            INSERT INTO chat_messages (message_id, session_id, role, content, token_usage, page_context)
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5)
            RETURNING message_id
            """,
            session_id,
            role,
            content,
            token_usage,
            page_context
        )

        return str(result[0]["message_id"]) if result else ""

    async def create_chat_session(
        self,
        user_agent: Optional[str] = None,
        current_page: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new chat session.

        Args:
            user_agent: User's browser info
            current_page: Initial page URL
            metadata: Additional session metadata

        Returns:
            Session ID
        """
        result = await self._execute(
            """
            INSERT INTO chat_sessions (session_id, user_agent, current_page, metadata)
            VALUES (gen_random_uuid(), $1, $2, $3)
            RETURNING session_id
            """,
            user_agent,
            current_page,
            json.dumps(metadata) if metadata else {}
        )

        return str(result[0]["session_id"]) if result else ""

    async def update_session_page(self, session_id: str, page_url: str):
        """Update the current page for a session."""
        await self._execute(
            """
            UPDATE chat_sessions
            SET current_page = $2, updated_at = NOW()
            WHERE session_id = $1
            """,
            session_id,
            page_url
        )

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        results = await self._execute(
            """
            SELECT session_id, created_at, updated_at, user_agent, current_page, metadata
            FROM chat_sessions
            WHERE session_id = $1
            """,
            session_id
        )

        if not results:
            return None

        r = results[0]
        return {
            "session_id": str(r["session_id"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            "user_agent": r["user_agent"],
            "current_page": r["current_page"],
            "metadata": r["metadata"]
        }

    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent chat sessions."""
        results = await self._execute(
            """
            SELECT session_id, created_at, updated_at, current_page
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT $1
            """,
            limit
        )

        return [
            {
                "session_id": str(r["session_id"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "current_page": r["current_page"]
            }
            for r in results
        ]


# Global server instance
_db_server: Optional[DatabaseMCPServer] = None


async def get_db_server() -> DatabaseMCPServer:
    """Get or create Database MCP server instance."""
    global _db_server
    if _db_server is None:
        _db_server = DatabaseMCPServer()
        await _db_server.connect()
    return _db_server


# Tool functions for agent

async def check_inventory(product_id: str) -> str:
    """
    Tool: Check product inventory status.

    Args:
        product_id: Product ID to check

    Returns:
        JSON string of stock information
    """
    server = await get_db_server()
    result = await server.check_product_stock(product_id)
    return json.dumps(result, indent=2)


async def get_chat_history_tool(session_id: str, limit: int = 50) -> str:
    """
    Tool: Get chat history for a session.

    Args:
        session_id: Session ID
        limit: Maximum messages

    Returns:
        JSON string of chat messages
    """
    server = await get_db_server()
    history = await server.get_chat_history(session_id, limit)
    return json.dumps(history, indent=2)


async def search_similar_content_tool(query: str, limit: int = 5) -> str:
    """
    Tool: Search for similar content using vector similarity.
    Note: This requires the query to be embedded first.

    Args:
        query: Search query text
        limit: Maximum results

    Returns:
        JSON string of similar content
    """
    # This will be called with pre-computed embeddings from the agent service
    # For now, return a placeholder
    return json.dumps({
        "error": "Use the embeddings service for vector search",
        "query": query
    }, indent=2)


async def save_message(
    session_id: str,
    role: str,
    content: str,
    token_usage: Optional[int] = None,
    page_context: Optional[str] = None
) -> str:
    """
    Tool: Save a chat message.

    Args:
        session_id: Session ID
        role: Message role
        content: Message content
        token_usage: Token count
        page_context: Page context

    Returns:
        JSON string with message ID
    """
    server = await get_db_server()
    message_id = await server.save_chat_message(
        session_id, role, content, token_usage, page_context
    )
    return json.dumps({"message_id": message_id, "success": True}, indent=2)


async def create_session(
    user_agent: Optional[str] = None,
    current_page: Optional[str] = None
) -> str:
    """
    Tool: Create a new chat session.

    Args:
        user_agent: Browser info
        current_page: Initial page

    Returns:
        JSON string with session ID
    """
    server = await get_db_server()
    session_id = await server.create_chat_session(user_agent, current_page)
    return json.dumps({"session_id": session_id, "success": True}, indent=2)


async def get_session(session_id: str) -> str:
    """
    Tool: Get session information.

    Args:
        session_id: Session ID

    Returns:
        JSON string of session info
    """
    server = await get_db_server()
    info = await server.get_session_info(session_id)
    if not info:
        return json.dumps({"error": "Session not found"})
    return json.dumps(info, indent=2)


# MCP Server runner for standalone execution

async def run_database_mcp_server(port: int = 3002):
    """Run the Database MCP server as HTTP server."""
    from aiohttp import web

    async def handle_check_stock(request):
        product_id = request.match_info["product_id"]
        result = await check_inventory(product_id)
        return web.json_response(json.loads(result))

    async def handle_get_history(request):
        session_id = request.match_info["session_id"]
        limit = int(request.query.get("limit", 50))
        result = await get_chat_history_tool(session_id, limit)
        return web.json_response(json.loads(result))

    async def handle_search(request):
        data = await request.json()
        query = data.get("query", "")
        limit = data.get("limit", 5)
        result = await search_similar_content_tool(query, limit)
        return web.json_response(json.loads(result))

    async def handle_save_message(request):
        data = await request.json()
        result = await save_message(
            data.get("session_id"),
            data.get("role"),
            data.get("content"),
            data.get("token_usage"),
            data.get("page_context")
        )
        return web.json_response(json.loads(result))

    async def handle_create_session(request):
        data = await request.json() if request.can_read_body else {}
        result = await create_session(
            data.get("user_agent"),
            data.get("current_page")
        )
        return web.json_response(json.loads(result))

    async def handle_get_session(request):
        session_id = request.match_info["session_id"]
        result = await get_session(session_id)
        return web.json_response(json.loads(result))

    app = web.Application()
    app.router.add_get("/stock/{product_id}", handle_check_stock)
    app.router.add_get("/history/{session_id}", handle_get_history)
    app.router.add_post("/search", handle_search)
    app.router.add_post("/message", handle_save_message)
    app.router.add_post("/session", handle_create_session)
    app.router.add_get("/session/{session_id}", handle_get_session)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()

    print(f"Database MCP Server running on http://localhost:{port}")
    print("Endpoints:")
    print(f"  GET  /stock/<product_id>")
    print(f"  GET  /history/<session_id>?limit=<limit>")
    print(f"  POST /search")
    print(f"  POST /message")
    print(f"  POST /session")
    print(f"  GET  /session/<session_id>")

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run_database_mcp_server())
