#!/usr/bin/env python3
"""
Newsletter Expert MCP Server

MCP server that provides newsletter querying capabilities to Cursor and other MCP clients.
"""

import os
import sys
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.models import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp import types

from src.rag import NewsletterRAG

# Initialize RAG system
rag = None


def initialize_rag():
    """Initialize the RAG system on server startup."""
    global rag
    try:
        rag = NewsletterRAG(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="newsletter_chunks"
        )
        print("✅ Newsletter RAG initialized", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}", file=sys.stderr)
        raise


# Create MCP server
server = Server("newsletter-expert")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="query_newsletters",
            description="Search and query newsletter emails using natural language. Returns relevant information from the user's newsletter collection with sources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the newsletters (e.g., 'What newsletters did I get about AI?')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant email chunks to retrieve (default: 5)",
                        "default": 5
                    }
                },
                "required": ["question"]
            }
        ),
        types.Tool(
            name="list_newsletter_stats",
            description="Get statistics about the indexed newsletter collection (number of chunks, collection status, etc.)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict[str, Any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution."""
    
    if not rag:
        return [types.TextContent(
            type="text",
            text="❌ RAG system not initialized. Please check server logs."
        )]
    
    if name == "query_newsletters":
        # Get arguments
        question = arguments.get("question", "")
        top_k = arguments.get("top_k", 5)
        
        if not question:
            return [types.TextContent(
                type="text",
                text="❌ Error: 'question' parameter is required"
            )]
        
        try:
            # Query the RAG system
            result = rag.query(question, top_k=top_k)
            
            # Format response
            answer = result["answer"]
            sources = result["sources"]
            
            # Build response with sources
            response_text = f"**Answer:**\n\n{answer}\n\n"
            response_text += f"**Sources ({len(sources)} newsletters found):**\n\n"
            
            for i, source in enumerate(sources, 1):
                response_text += f"{i}. **{source['subject']}**\n"
                response_text += f"   From: {source['from']}\n"
                response_text += f"   Date: {source['date']}\n"
                response_text += f"   Relevance: {source['score']:.3f}\n"
                response_text += f"   Preview: {source['text_preview']}\n\n"
            
            return [types.TextContent(
                type="text",
                text=response_text
            )]
        
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error querying newsletters: {str(e)}"
            )]
    
    elif name == "list_newsletter_stats":
        try:
            # Get collection info from Qdrant
            collection_info = rag.qdrant_client.get_collection(rag.collection_name)
            
            stats_text = f"**Newsletter Collection Statistics:**\n\n"
            stats_text += f"- Collection: `{rag.collection_name}`\n"
            stats_text += f"- Total chunks indexed: {collection_info.points_count}\n"
            stats_text += f"- Vector dimensions: {collection_info.config.params.vectors.size}\n"
            stats_text += f"- Distance metric: {collection_info.config.params.vectors.distance}\n"
            stats_text += f"- Status: {collection_info.status}\n"
            
            return [types.TextContent(
                type="text",
                text=stats_text
            )]
        
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error getting stats: {str(e)}"
            )]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"❌ Unknown tool: {name}"
        )]


async def main():
    """Run the MCP server."""
    # Initialize RAG system
    initialize_rag()
    
    # Start stdio server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="newsletter-expert",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
