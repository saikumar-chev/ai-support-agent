"""
Intent Routing for the AI Support Agent.

This module is the core decision-making component of the agent. It analyzes
the user's query and routes it to the appropriate handler (RAG, Tool, or a
combination).

The primary function, `route_query`, uses a deterministic, rule-based approach
to classify the user's intent. This method is efficient and predictable. It
first checks for keywords and patterns (like order IDs) to categorize the query
into predefined intents.

The intents are:
- KNOWLEDGE: General questions answered by the RAG system.
- ORDER_STATUS: Questions about the status of a specific order.
- ORDER_DETAILS: Questions about the contents of a specific order.
- RETURN_ELIGIBILITY: A complex query combining a tool call (get order info)
  and a RAG call (get return policy).
- UNKNOWN: Queries that don't fit any other category.
"""

import logging
import re
from enum import Enum

from utils import setup_logging

# Configure logging
logger = setup_logging()

class Intent(Enum):
    """Enumeration for user query intents."""
    KNOWLEDGE = "knowledge_base_query"
    ORDER_STATUS = "get_order_status"
    ORDER_DETAILS = "get_order_details"
    RETURN_ELIGIBILITY = "check_return_eligibility"
    UNKNOWN = "unknown_intent"

# Regex to find order IDs like 'ORD1001'
ORDER_ID_PATTERN = re.compile(r'\b(ORD\d{4})\b', re.IGNORECASE)

def extract_order_id(query: str) -> str | None:
    """
    Extracts the first matching order ID from a query using regex.

    Args:
        query: The user's input string.

    Returns:
        The extracted order ID (e.g., 'ORD1001') or None if not found.
    """
    match = ORDER_ID_PATTERN.search(query)
    if match:
        order_id = match.group(1).upper()
        logger.info(f"Extracted order ID: {order_id}")
        return order_id
    return None

def route_query(query: str) -> tuple[Intent, str | None]:
    """
    Classifies a user query into an intent and extracts an order ID if present.

    Args:
        query: The user's input string.

    Returns:
        A tuple containing the classified Intent and the extracted order ID (or None).
    """
    query_lower = query.lower()
    order_id = extract_order_id(query)

    # ----------------------------
    # Order-related queries
    # ----------------------------
    if order_id:
        if "return" in query_lower or "exchange" in query_lower:
            logger.info(f"Routing query to: RETURN_ELIGIBILITY for order {order_id}")
            return Intent.RETURN_ELIGIBILITY, order_id

        if (
            "status" in query_lower
            or "where is" in query_lower
            or "track" in query_lower
        ):
            logger.info(f"Routing query to: ORDER_STATUS for order {order_id}")
            return Intent.ORDER_STATUS, order_id

        logger.info(f"Routing query to: ORDER_DETAILS for order {order_id}")
        return Intent.ORDER_DETAILS, order_id

    # --------------------------------------------------------------------
    # If no order ID is found, default to a knowledge base query.
    # This is more robust than relying on a fixed list of keywords.
    # --------------------------------------------------------------------
    if not order_id:
        logger.info("Routing query to: KNOWLEDGE")
        return Intent.KNOWLEDGE, None

    # ----------------------------
    # Unknown queries
    # ----------------------------
    logger.warning(f"Could not determine intent for query: '{query}'")
    return Intent.UNKNOWN, None