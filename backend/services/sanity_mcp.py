"""
Sanity MCP Server - Provides tools for querying Sanity CMS products.
Phase 12: MCP Server Development
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx

# Configuration
SANITY_PROJECT_ID = os.getenv("NEXT_PUBLIC_SANITY_PROJECT_ID", "")
SANITY_DATASET = os.getenv("NEXT_PUBLIC_SANITY_DATASET", "production")
SANITY_API_TOKEN = os.getenv("SANITY_API_TOKEN", "")
SANITY_API_VERSION = "2024-01-01"

SANITY_BASE_URL = f"https://{SANITY_PROJECT_ID}.api.sanity.io/v{ SANITY_API_VERSION}"


class SanityMCPServer:
    """MCP Server for Sanity CMS product queries."""

    def __init__(self):
        self.base_url = SANITY_BASE_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SANITY_API_TOKEN}"
        } if SANITY_API_TOKEN else {}

    async def _fetch(self, query: str, params: Dict = None) -> Any:
        """Execute GROQ query against Sanity API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/data/query/{SANITY_DATASET}",
                headers=self.headers,
                json={
                    "query": query,
                    "params": params or {}
                },
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("result", [])
            else:
                print(f"Sanity API error: {response.status_code} - {response.text}")
                return []

    async def search_products_by_name(
        self,
        name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search products by name.

        Args:
            name: Product name to search for
            limit: Maximum number of results

        Returns:
            List of matching products
        """
        query = """
        *[_type == "product" && name match $name] | order(_score desc) [0...$limit] {
            _id,
            name,
            "slug": slug.current,
            shortDescription,
            price,
            compareAtPrice,
            "category": category->name,
            "imageUrl": mainImage.asset->url,
            stockStatus,
            featured
        }
        """
        return await self._fetch(query, {"name": f"{name}*", "limit": limit})

    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full product details by ID.

        Args:
            product_id: Sanity document ID

        Returns:
            Product details or None
        """
        query = """
        *[_type == "product" && _id == $id][0] {
            _id,
            _type,
            name,
            "slug": slug.current,
            description,
            shortDescription,
            price,
            compareAtPrice,
            "category": category->{
                _id,
                name,
                "slug": slug.current
            },
            mainImage{
                asset->{
                    _id,
                    url
                },
                alt
            },
            images[]{
                asset->{
                    _id,
                    url
                },
                alt
            },
            dimensions,
            materials,
            colors,
            stockStatus,
            sku,
            weight,
            warranty,
            careInstructions,
            featured
        }
        """
        result = await self._fetch(query, {"id": product_id})
        return result[0] if result else None

    async def get_product_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Get full product details by slug.

        Args:
            slug: Product slug

        Returns:
            Product details or None
        """
        query = """
        *[_type == "product" && slug.current == $slug][0] {
            _id,
            _type,
            name,
            "slug": slug.current,
            description,
            shortDescription,
            price,
            compareAtPrice,
            "category": category->{
                _id,
                name,
                "slug": slug.current
            },
            mainImage{
                asset->{
                    _id,
                    url
                },
                alt
            },
            images[]{
                asset->{
                    _id,
                    url
                },
                alt
            },
            dimensions,
            materials,
            colors,
            stockStatus,
            sku,
            weight,
            warranty,
            careInstructions,
            featured
        }
        """
        result = await self._fetch(query, {"slug": slug})
        return result[0] if result else None

    async def get_products_by_category(
        self,
        category_name: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get products by category name.

        Args:
            category_name: Category name to filter by
            limit: Maximum number of results

        Returns:
            List of products in category
        """
        query = """
        *[_type == "product" && category->name == $category] | order(_createdAt desc) [0...$limit] {
            _id,
            name,
            "slug": slug.current,
            shortDescription,
            price,
            compareAtPrice,
            "category": category->name,
            "imageUrl": mainImage.asset->url,
            stockStatus,
            featured
        }
        """
        return await self._fetch(query, {"category": category_name, "limit": limit})

    async def search_products_by_filter(
        self,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        materials: Optional[List[str]] = None,
        in_stock_only: bool = False,
        featured_only: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Advanced product search with filters.

        Args:
            category: Category name filter
            min_price: Minimum price filter
            max_price: Maximum price filter
            materials: List of materials to filter
            in_stock_only: Only show in-stock items
            featured_only: Only show featured items
            limit: Maximum number of results

        Returns:
            List of matching products
        """
        # Build filter conditions
        conditions = ['_type == "product"']

        if category:
            conditions.append('category->name == $category')

        if min_price is not None:
            conditions.append('price >= $minPrice')

        if max_price is not None:
            conditions.append('price <= $maxPrice')

        if in_stock_only:
            conditions.append('stockStatus == "in-stock"')

        if featured_only:
            conditions.append('featured == true')

        filter_query = " && ".join(conditions)

        query = f"""
        *[{filter_query}] | order(_createdAt desc) [0...$limit] {{
            _id,
            name,
            "slug": slug.current,
            shortDescription,
            price,
            compareAtPrice,
            "category": category->name,
            "imageUrl": mainImage.asset->url,
            stockStatus,
            featured,
            materials,
            dimensions
        }}
        """

        params = {"limit": limit}
        if category:
            params["category"] = category
        if min_price is not None:
            params["minPrice"] = min_price
        if max_price is not None:
            params["maxPrice"] = max_price

        return await self._fetch(query, params)

    async def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all product categories."""
        query = """
        *[_type == "category"] | order(name asc) {
            _id,
            name,
            "slug": slug.current,
            description,
            "imageUrl": image.asset->url
        }
        """
        return await self._fetch(query)

    async def get_featured_products(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Get featured products."""
        query = """
        *[_type == "product" && featured == true] | order(_createdAt desc) [0...$limit] {
            _id,
            name,
            "slug": slug.current,
            shortDescription,
            price,
            compareAtPrice,
            "category": category->name,
            "imageUrl": mainImage.asset->url,
            stockStatus,
            featured
        }
        """
        return await self._fetch(query, {"limit": limit})


# Global server instance
_sanity_server: Optional[SanityMCPServer] = None


async def get_sanity_server() -> SanityMCPServer:
    """Get or create Sanity MCP server instance."""
    global _sanity_server
    if _sanity_server is None:
        _sanity_server = SanityMCPServer()
    return _sanity_server


# Tool functions that can be called by the agent

async def search_products(query: str, category: Optional[str] = None) -> str:
    """
    Tool: Search products by name or description.

    Args:
        query: Product name or search term
        category: Optional category to filter by

    Returns:
        JSON string of matching products
    """
    server = await get_sanity_server()

    if category:
        products = await server.get_products_by_category(category, limit=10)
    else:
        products = await server.search_products_by_name(query, limit=10)

    return json.dumps(products, indent=2)


async def get_product_details(product_id: str) -> str:
    """
    Tool: Get full product details by ID.

    Args:
        product_id: Sanity product document ID

    Returns:
        JSON string of product details
    """
    server = await get_sanity_server()
    product = await server.get_product_by_id(product_id)

    if not product:
        return json.dumps({"error": "Product not found"})

    return json.dumps(product, indent=2)


async def get_product_by_slug_tool(slug: str) -> str:
    """
    Tool: Get full product details by slug.

    Args:
        slug: Product URL slug

    Returns:
        JSON string of product details
    """
    server = await get_sanity_server()
    product = await server.get_product_by_slug(slug)

    if not product:
        return json.dumps({"error": "Product not found"})

    return json.dumps(product, indent=2)


async def get_products_by_category_tool(category: str, limit: int = 20) -> str:
    """
    Tool: Get products by category name.

    Args:
        category: Category name
        limit: Maximum results

    Returns:
        JSON string of products
    """
    server = await get_sanity_server()
    products = await server.get_products_by_category(category, limit=limit)

    return json.dumps(products, indent=2)


async def search_products_filtered(
    query: str = "",
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = False,
    featured_only: bool = False,
    limit: int = 20
) -> str:
    """
    Tool: Search products with advanced filters.

    Args:
        query: Search term (optional)
        category: Category filter
        min_price: Minimum price
        max_price: Maximum price
        in_stock_only: Filter in-stock only
        featured_only: Filter featured only
        limit: Maximum results

    Returns:
        JSON string of matching products
    """
    server = await get_sanity_server()
    products = await server.search_products_by_filter(
        category=category,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
        featured_only=featured_only,
        limit=limit
    )

    return json.dumps(products, indent=2)


async def get_categories() -> str:
    """
    Tool: Get all product categories.

    Returns:
        JSON string of categories
    """
    server = await get_sanity_server()
    categories = await server.get_all_categories()

    return json.dumps(categories, indent=2)


async def get_featured(limit: int = 6) -> str:
    """
    Tool: Get featured products.

    Args:
        limit: Maximum number of results

    Returns:
        JSON string of featured products
    """
    server = await get_sanity_server()
    products = await server.get_featured_products(limit=limit)

    return json.dumps(products, indent=2)


# MCP Server runner for standalone execution

async def run_sanity_mcp_server(port: int = 3001):
    """Run the Sanity MCP server as HTTP server."""
    from aiohttp import web

    async def handle_search_products(request):
        query = request.query.get("q", "")
        category = request.query.get("category")
        result = await search_products(query, category)
        return web.json_response(json.loads(result))

    async def handle_get_product(request):
        product_id = request.match_info["product_id"]
        result = await get_product_details(product_id)
        return web.json_response(json.loads(result))

    async def handle_get_by_slug(request):
        slug = request.match_info["slug"]
        result = await get_product_by_slug_tool(slug)
        return web.json_response(json.loads(result))

    async def handle_get_category(request):
        category = request.match_info["category"]
        limit = int(request.query.get("limit", 20))
        result = await get_products_by_category_tool(category, limit)
        return web.json_response(json.loads(result))

    async def handle_search_filtered(request):
        data = await request.json()
        result = await search_products_filtered(**data)
        return web.json_response(json.loads(result))

    async def handle_categories(request):
        result = await get_categories()
        return web.json_response(json.loads(result))

    async def handle_featured(request):
        limit = int(request.query.get("limit", 6))
        result = await get_featured(limit)
        return web.json_response(json.loads(result))

    app = web.Application()
    app.router.add_get("/search", handle_search_products)
    app.router.add_get("/product/{product_id}", handle_get_product)
    app.router.add_get("/product/slug/{slug}", handle_get_by_slug)
    app.router.add_get("/category/{category}", handle_get_category)
    app.router.add_post("/search/filter", handle_search_filtered)
    app.router.add_get("/categories", handle_categories)
    app.router.add_get("/featured", handle_featured)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()

    print(f"Sanity MCP Server running on http://localhost:{port}")
    print("Endpoints:")
    print(f"  GET  /search?q=<query>&category=<category>")
    print(f"  GET  /product/<product_id>")
    print(f"  GET  /product/slug/<slug>")
    print(f"  GET  /category/<category>")
    print(f"  POST /search/filter")
    print(f"  GET  /categories")
    print(f"  GET  /featured")

    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run_sanity_mcp_server())
