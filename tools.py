"""
Tools for the AI Support Agent to interact with the order database.

This module defines a set of deterministic functions that can be used to
query order information from a CSV file. These functions are designed to be
used as "tools" by a LangChain agent.

Each function is decorated with `@tool` to make it discoverable by the agent.
They include detailed docstrings explaining their purpose, arguments, and
return values, which the LLM uses to decide when to call them.

The functions are robust, handling cases where an order ID is not found
and returning a clear, user-friendly message.
"""

import logging
import pandas as pd
from langchain.tools import tool
from config import ORDERS_CSV_PATH
from utils import setup_logging

# Configure logging
logger = setup_logging()

def _load_orders_data() -> pd.DataFrame:
    """
    Loads order data from the CSV file.

    This private helper function reads the orders CSV into a pandas DataFrame.
    It includes error handling for cases where the file might not be found.

    Returns:
        pd.DataFrame: A DataFrame containing the order data, or an empty
                      DataFrame if the file is not found.
    """
    try:
        df = pd.read_csv(ORDERS_CSV_PATH)
        logger.info(f"Successfully loaded order data from {ORDERS_CSV_PATH}")
        return df
    except FileNotFoundError:
        logger.error(f"Order data file not found at: {ORDERS_CSV_PATH}")
        return pd.DataFrame()

# Load the data once when the module is imported for efficiency
_orders_df = _load_orders_data()

@tool
def get_order_status(order_id: str) -> str:
    """
    Retrieves the current status of a specific order.

    Args:
        order_id: The unique identifier for the order (e.g., 'ORD1001').

    Returns:
        The status of the order (e.g., 'Shipped', 'Processing', 'Delivered')
        or a message indicating the order was not found.
    """
    logger.info(f"Tool 'get_order_status' called for order_id: {order_id}")
    if _orders_df.empty:
        return "Error: Order data is not available."

    order = _orders_df[_orders_df['order_id'] == order_id]
    if not order.empty:
        status = order['status'].iloc[0]
        return f"The status of order {order_id} is: {status}."
    else:
        return f"Order with ID '{order_id}' not found."

@tool
def get_order_details(order_id: str) -> str:
    """
    Provides a summary of a specific order, including product, quantity, and total price.

    Args:
        order_id: The unique identifier for the order (e.g., 'ORD1001').

    Returns:
        A summary of the order details or a message indicating the order was not found.
    """
    logger.info(f"Tool 'get_order_details' called for order_id: {order_id}")
    if _orders_df.empty:
        return "Error: Order data is not available."

    order = _orders_df[_orders_df['order_id'] == order_id]
    if not order.empty:
        details = order.iloc[0]
        return (
            f"Order {order_id} details: "
            f"Product: {details['product_name']}, "
            f"Quantity: {details['quantity']}, "
            f"Total Price: ${details['total_price']:.2f}, "
            f"Status: {details['status']}."
        )
    else:
        return f"Order with ID '{order_id}' not found."

# A list of all available tools for easy import into the router
available_tools = [get_order_status, get_order_details]

if __name__ == '__main__':
    # Example usage for testing the tools directly
    print("Testing tools...")
    print(get_order_status("ORD1004"))
    print(get_order_status("ORD9999")) # Test non-existent order
    print(get_order_details("ORD1001"))
    print(get_order_details("ORD1002"))