"""
Retrieval-Augmented Generation (RAG) System for the AI Support Agent.

This module encapsulates the logic for the RAG pipeline, which is responsible
for answering knowledge-based questions. It handles the following steps:
1.  Loading knowledge base documents from a specified directory.
2.  Splitting the documents into manageable chunks.
3.  Generating embeddings for these chunks using a selected model.
4.  Storing the chunks and their embeddings in a FAISS vector store.
5.  Providing a retrieval mechanism to find relevant documents for a given query.
6.  Creating a LangChain chain that combines retrieval and language model
    generation to produce a final answer.

The system is designed to be modular, using configuration from `config.py`
and supporting different LLM and embedding providers. It also persists the
vector store to disk to avoid costly re-indexing on every run.
"""

import logging
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from utils import setup_logging

# Configure logging
logger = setup_logging()

class RAGSystem:
    """
    Encapsulates the entire RAG pipeline from document loading to answer generation.
    """

    def __init__(self):
        """
        Initializes the RAG system by setting up models, embeddings, and the vector store.
        """
        logger.info("Initializing RAG System...")
        self._initialize_models()
        self._initialize_vector_store()

    def _initialize_models(self):
        """Selects and initializes the LLM and embedding models based on the provider."""
        if config.LLM_PROVIDER == "google":
            logger.info("Using Google (Gemini) models.")
            # Let LangChain read the API key from the environment variables
            self.llm = GoogleGenerativeAI(model=config.GEMINI_MODEL)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=config.GEMINI_EMBEDDING_MODEL)
        elif config.LLM_PROVIDER == "openai":
            logger.info("Using OpenAI models.")
            # Let LangChain read the API key from the environment variables
            self.llm = ChatOpenAI(model=config.OPENAI_MODEL)
            self.embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}. Please choose 'openai' or 'google'.")

    def _initialize_vector_store(self):
        """
        Loads the vector store from disk if it exists, otherwise creates it from
        the source documents.
        """
        vectorstore_path = config.VECTORSTORE_DIR
        index_file = vectorstore_path / "index.faiss"

        # More robust check: ensure the main index file exists.
        if index_file.exists():
            logger.info(f"Loading existing vector store from {vectorstore_path}")
            self.vectorstore = FAISS.load_local(str(vectorstore_path), self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully.")
        else:
            logger.info("No existing vector store found. Creating a new one.")
            self.vectorstore = self._create_vector_store(vectorstore_path)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    def _create_vector_store(self, path: Path) -> FAISS:
        """
        Creates a new FAISS vector store from the documents in the docs directory.

        Returns:
            FAISS: The newly created vector store instance.
        """
        logger.info(f"Loading documents from {config.DOCS_DIR}...")
        loader = DirectoryLoader(
            str(config.DOCS_DIR),  # Convert Path object to string
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks.")

        logger.info("Generating embeddings and creating FAISS index...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info("FAISS index created.")

        # Save the vector store to disk for future use
        vectorstore.save_local(str(path))
        logger.info(f"Vector store saved to {config.VECTORSTORE_DIR}")

        return vectorstore

    def get_rag_chain(self):
        """
        Builds and returns the runnable RAG chain.

        The chain performs the following steps:
        1. Retrieves context relevant to the user's question.
        2. Formats the retrieved context and question into a prompt.
        3. Passes the prompt to the LLM.
        4. Parses the LLM's output into a string.

        Returns:
            A LangChain runnable sequence.
        """
        template = """
        You are a helpful and polite customer support agent.
        Answer the user's question based only on the following context.
        If the information is not in the context, politely say that you cannot find the answer.
        Do not make up information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

if __name__ == '__main__':
    # Example usage for testing the RAG system directly
    rag_system = RAGSystem()
    chain = rag_system.get_rag_chain()
    print("\n--- Testing RAG Chain ---")
    question = "What is your return policy for electronics?"
    print(f"Question: {question}")
    response = chain.invoke(question)
    print(f"Response: {response}")