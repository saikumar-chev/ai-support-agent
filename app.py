"""
Main application file for the AI Support Agent.

This script serves as the entry point for the command-line interface (CLI)
of the support agent. It orchestrates the entire workflow:
1.  Initializes all necessary components: logging, RAG system, and tools.
2.  Enters a loop to accept user queries from the command line.
3.  Uses the `router` module to determine the intent of each query.
4.  Based on the intent, it calls the appropriate handler:
    - The RAG chain for knowledge-based questions.
    - A specific tool for order data questions.
    - A combination of a tool and the RAG chain for complex queries.
5.  Prints the agent's response to the console.

The application is designed to be modular and extensible, with clear separation
between the UI (this file), the routing logic, and the core AI/data capabilities.
"""

import logging
from router import route_query, Intent
from rag import RAGSystem
from tools import get_order_status, get_order_details
from utils import setup_logging

# Configure logging
logger = setup_logging()

class SupportAgent:
    """
    The main class for the AI Support Agent.
    """

    def __init__(self):
        """
        Initializes the agent, loading the RAG system and defining tool mappings.
        """
        logger.info("Initializing Support Agent...")
        self.rag_system = RAGSystem()
        self.rag_chain = self.rag_system.get_rag_chain()
        self.tools = {
            Intent.ORDER_STATUS: get_order_status,
            Intent.ORDER_DETAILS: get_order_details,
        }
        logger.info("Support Agent initialized successfully.")

    def handle_query(self, query: str):
        """
        Handles a single user query by routing and executing it.

        Args:
            query: The user's input query.
        """
        intent, order_id = route_query(query)

        if intent == Intent.KNOWLEDGE:
            response = self.rag_chain.invoke(query)
        elif intent in self.tools:
            tool_function = self.tools[intent]
            response = tool_function.invoke({"order_id": order_id})
        elif intent == Intent.RETURN_ELIGIBILITY:
            # The router ensures order_id is a string for this intent.
            # We add a check for type safety, though it's unlikely to be None.
            if order_id: 
                response = self._handle_return_eligibility(query, order_id)
            else:
                # This case should ideally not be reached if router logic is correct,
                # but we handle it for robustness and to satisfy the type checker.
                response = "Could not find an order ID in your request. Please specify an order ID to check for return eligibility."
        else:  # Intent.UNKNOWN
            response = "I'm sorry, I'm not sure how to help with that. Could you please rephrase your question?"

        print(f"\nAgent:\n{response}\n")

    def _handle_return_eligibility(self, query: str, order_id: str) -> str:
        """
        Handles the complex 'return eligibility' query.

        This function first calls a tool to get order details and then uses
        that information to augment the query to the RAG system.

        Args:
            query: The original user query.
            order_id: The extracted order ID.

        Returns:
            A comprehensive answer about the return eligibility.
        """
        logger.info(f"Handling combined query for return eligibility of order {order_id}")

        # Use the .invoke() method to call the LangChain tool.
        order_details = get_order_details.invoke({"order_id": order_id})

        # Check if the order was found
        if "not found" in order_details.lower() or "error" in order_details.lower():
            return order_details

        # Augment the original query with the retrieved order details
        augmented_query = (
            f"Based on the following order details, can this order be returned? "
            f"Order Details: {order_details}. "
            f"Original question: {query}"
        )
        logger.info(f"Augmented query for RAG system: {augmented_query}")

        # Call the RAG chain with the augmented query
        return self.rag_chain.invoke(augmented_query)

    def run(self):
        """
        Starts the main command-line interface loop for the agent.
        """
        print("AI Support Agent is ready. Type 'quit' or 'exit' to stop.")
        while True:
            user_query = input("You:\n")
            if user_query.lower() in ["quit", "exit"]:
                print("Agent:\nGoodbye!")
                break
            if not user_query.strip():
                continue
            self.handle_query(user_query)

if __name__ == "__main__":
    try:
        agent = SupportAgent()
        agent.run()
    except Exception as e:
        logger.error(f"Failed to initialize and run the Support Agent: {e}", exc_info=True)
        print("\n[ERROR] The application failed to start. Please check the following:")
        print("1. Ensure your GOOGLE_API_KEY in the .env file is correct.")
        print("2. Verify that all dependencies are installed (`pip install -r requirements.txt`).")
        print("3. Check the `app.log` file for detailed error information.")