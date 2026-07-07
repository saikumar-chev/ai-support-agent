# AI Mini Support Agent for E-commerce

This project is a production-quality AI support agent designed for an e-commerce seller. It uses a combination of Retrieval-Augmented Generation (RAG) and deterministic tool-calling to answer customer queries about company policies and specific order information.

## Project Overview

The agent can handle three types of questions:

1.  **Knowledge Questions**: Answers questions about company policies (shipping, returns, etc.) by retrieving information from Markdown documents using a RAG pipeline.
2.  **Data Questions**: Answers questions about specific orders (e.g., status, details) by looking up data in a `orders.csv` file using deterministic Python tools.
3.  **Combined Questions**: Handles complex queries (e.g., "Can I return order ORD1004?") by first calling a tool to get order data and then using that data to inform a RAG query.

The core of the system is an intent router that intelligently decides which approach to use based on the user's query.

## Architecture Diagram

The application follows a modular architecture that separates concerns, making it easy to maintain and extend.

```
   User Query
       │
       ▼
┌──────────────┐
│   app.py     │
│ (CLI)        │
└──────────────┘
       │
       ▼
┌──────────────┐
│  router.py   │
│(Intent Router)│
└──────────────┘
       │
       ├─► Intent: KNOWLEDGE ───────────► ┌───────────┐
       │                                  │  rag.py   │
       │                                  │ (RAG Chain) │
       │                                  └─────┬─────┘
       │                                        │
       │                                  ┌─────▼─────┐
       │                                  │ vectorstore/│
       │                                  │  (FAISS)    │
       │                                  └─────────────┘
       │
       ├─► Intent: ORDER_STATUS/DETAILS ► ┌───────────┐
       │                                  │ tools.py  │
       │                                  │(CSV Tools)│
       │                                  └─────┬─────┘
       │                                        │
       │                                  ┌─────▼─────┐
       │                                  │orders.csv │
       │                                  └─────────────┘
       │
       └─► Intent: RETURN_ELIGIBILITY ──► tools.py then rag.py
```

## Folder Structure

```
ai-support-agent/
├── .env.example
├── README.md
├── requirements.txt
├── app.py
├── config.py
├── rag.py
├── router.py
├── tools.py
├── utils.py
├── sample_data/
│   ├── docs/
│   │   ├── account_and_support.md
│   │   ├── payment_and_pricing.md
│   │   ├── returns_and_refunds.md
│   │   └── shipping_policy.md
│   └── orders.csv
├── tests/
│   └── test_cases.md
└── vectorstore/
    └── .gitkeep
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ai-support-agent
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Environment Variables

The application requires API keys for the chosen LLM provider.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

2.  Open the `.env` file and add your API keys:
    ```
    # OpenAI API Key
    OPENAI_API_KEY="your_openai_api_key_here"

    # Google Gemini API Key
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

3.  You can choose the LLM provider by setting the `LLM_PROVIDER` variable in `config.py` to either `"openai"` or `"google"`.

## How to Run

1.  **First Run:** The first time you run the application, it will automatically create the vector store from the documents in `sample_data/docs/`.

2.  **Start the agent:**
    ```bash
    python app.py
    ```

The command-line interface will start, and you can begin asking questions.

## Technical Decisions

### Routing Logic

The system uses a **deterministic, rule-based router** (`router.py`) as the first line of classification.
- It uses a regular expression (`\b(ORD\d{4})\b`) to detect the presence of an order ID.
- If an order ID is found, it checks for keywords like "return" or "status" to route to the appropriate tool or combined workflow.
- If no order ID is found, it defaults to a `KNOWLEDGE` intent, which is handled by the RAG system.
- This approach is fast, predictable, and cost-effective, avoiding LLM calls for simple classification tasks.

### Chunking Strategy

- **Strategy**: `RecursiveCharacterTextSplitter`
- **Chunk Size**: 1000 characters (`CHUNK_SIZE`)
- **Chunk Overlap**: 150 characters (`CHUNK_OVERLAP`)

This strategy splits text recursively by different separators (`\n\n`, `\n`, ` `) to keep related pieces of text together. The overlap ensures that semantic context is not lost at the boundary of a split.

### Embedding Model

The system is configured to use either:
- **OpenAI**: `text-embedding-3-small`
- **Google**: `models/embedding-001`

These models were chosen for their strong performance and cost-effectiveness. The choice is easily configurable in `config.py`.

### Vector Database Choice

**FAISS (Facebook AI Similarity Search)** was chosen for this project due to its:
- **Simplicity**: It's a library, not a managed service, making local setup trivial.
- **Performance**: It is highly optimized for fast similarity searches in memory.
- **Cost**: It runs locally, incurring no additional costs.
- **Persistence**: The FAISS index is saved to the `vectorstore/` directory, so embeddings are only generated once.

## Trade-offs and Future Improvements

### Trade-offs

- **Deterministic vs. LLM Routing**: The current rule-based router is efficient but less flexible than an LLM-based router. It may fail on queries with complex phrasing that don't match the predefined keywords.
- **Local vs. Managed Vector DB**: FAISS is excellent for local development but does not scale for a multi-user, production application. A managed database like Pinecone, Weaviate, or ChromaDB would be a better choice for a real-world deployment.

### Future Improvements

- **Advanced Routing**: Implement a hybrid router that uses the deterministic router first and falls back to an LLM-based router for `UNKNOWN` intents.
- **Asynchronous Operations**: For a web-based UI, tool and RAG calls should be made asynchronous to avoid blocking the server.
- **More Tools**: Add more tools, such as `cancel_order(order_id)` or `get_customer_details(customer_id)`.
- **Web Interface**: Build a simple web UI using Flask or FastAPI to make the agent more accessible.
- **Automated Testing**: Add a suite of unit and integration tests using `pytest`.