import os
import uuid
from typing import List, Dict, Any, Optional
from functools import partial
from .core.state import InputRequest, ConversationState
from .retriever.search import VectorDB, HybridRetriever
from .core.agent.graph import build_conversation_workflow
from .core.nodes import retrieve_documents_node, transform_question, validate_documents_with_crag, generate_response
from .config import INGEST_URLS, INGEST_TOPICS 






class RagCLI:
    """Command-line interface for Stock Research."""

    def __init__(self):
        """Initialize the CLI."""
        # The vector database instance, initially empty.
        self.vector_db = None
        # A list to hold the processed and chunked documents.
        self.docs = None
        # The system prompt that guides the AI's behavior.
        self.system_prompt = None
        # The compiled LangGraph workflow for processing conversations.
        self.conversation_flow = None
        # A unique ID to track the current conversation session.
        self.thread_id = str(uuid.uuid4())
        # A list to store the history of user and assistant messages.
        self.conversation_history = []

    def initialize(self):
        """Initialize the system with documents."""
        try:
            print("Initializing system with documents...")

            # Define and create the directory for storing source documents.
            # This path is relative to the current file's location.
            documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documents")
            os.makedirs(documents_dir, exist_ok=True)
            print(documents_dir)
            
            # Define the initial data sources for ingestion.
            # In a real application, this might be user-provided.
            inputs = InputRequest(
                urls=INGEST_URLS,
                topics=INGEST_TOPICS,
                directory=documents_dir
            )

            # Perform the data ingestion and vectorization process.
            ingestion_result = self.perform_ingestion(inputs)

            # Store the results of the ingestion process in the instance.
            self.docs = ingestion_result["docs"]
            self.vector_db = ingestion_result["vector_db"]
            self.system_prompt = ingestion_result["system_prompt"]

            # Prepare the document retrieval node for the graph.
            # functools.partial pre-fills the `vector_db` argument, making the
            # node a pure function of state, which is required by the graph.
            ready_retrieve_node = partial(retrieve_documents_node, vector_db=self.vector_db)

            # Build the conversational graph, injecting the ready-to-use retrieve node.
            self.conversation_flow = build_conversation_workflow(retrieve_node=ready_retrieve_node)

            print(f"Initialized with {len(self.docs)} documents")
            print("System is ready for your questions!")
            print("Type 'exit' to quit, 'help' for commands")
            return True
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
            return False

    def perform_ingestion(self, inputs: InputRequest) -> dict:
        """Orchestrates the entire document ingestion and processing pipeline."""
        # Local imports are used here, possibly to avoid circular dependencies
        # or to scope them to this specific, heavy operation.
        from .config import slm, llm
        from .retriever.search import (
            load_documents_from_directory,
            process_urls,
            process_topics,
            chunk_documents
        )

        # Aggregate documents from all specified sources.
        all_docs = []
        if inputs.directory:
            all_docs.extend(load_documents_from_directory(inputs.directory))
        if inputs.urls:
            all_docs.extend(process_urls(inputs.urls))
        if inputs.topics:
            all_docs.extend(process_topics(inputs.topics))

        # Split the loaded documents into smaller, manageable chunks.
        chunked_docs = chunk_documents(all_docs)
        # Initialize the vector database.
        vector_db = VectorDB()
        # Add the chunked documents to the database to create embeddings.
        vector_db.add_documents(chunked_docs)

        # Create a context from the documents to generate a system prompt.
        context = "\n\n".join(doc.page_content for doc in chunked_docs)
        # This prompt asks a smaller language model (slm) to create a persona
        # and instructions for the main AI, based on the provided content.
        prompt = f"""Based on the excerpts below, generate a prompt that defines how an assistant should behave when answering questions about this content:

{context}

Instructions:
- Describe the assistant's area of expertise.
- Explain how it should respond to questions.
- Be concise and direct.
- Explicitly state that its primary source of information is the retrieved documents.
- Remind it to reply naturally, as if in a friendly company conversation.
"""
        generated_prompt = llm.invoke(prompt).content.strip()
        # Return all the artifacts from the ingestion process.
        return {
            "docs": chunked_docs,
            "vector_db": vector_db,
            "system_prompt": generated_prompt
        }

    def process_query(self, question: str) -> str:
        """Process a query from the user."""
        # Guard clause to ensure the system has been initialized.
        if not self.vector_db or not self.docs or not self.conversation_flow:
            return "System not initialized. Please wait..."

        try:
            # Prepare the state dictionary to be passed to the graph.
            # This represents the current state of the conversation.
            conversation_state = {
                "docs": self.docs,
                "current_question": question,
                "system_prompt": self.system_prompt,
                "conversation_history": self.conversation_history,
            }

            # Configure the graph invocation with the session's thread_id
            # to maintain stateful conversation history.
            config = {"configurable": {"thread_id": self.thread_id}}
            result = self.conversation_flow.invoke(conversation_state, config=config)

            # Extract the response and updated history from the graph's result.
            response = result.get("response", "No response generated")
            self.conversation_history = result.get("conversation_history", self.conversation_history)

            return response
        except Exception as e:
            # Provide detailed error information if something goes wrong.
            import traceback
            error_details = traceback.format_exc()
            return f"Error processing query: {str(e)}\n{error_details}"

    def run(self):
        """Run the CLI in a loop."""
        print("Welcome to Stock Research CLI!")
        print("Initializing system...")

        # Start the system; if initialization fails, exit gracefully.
        if not self.initialize():
            print("Failed to initialize system. Exiting...")
            return

        # Main event loop to continuously accept user input.
        while True:
            try:
                user_input = input("\n> ").strip()

                # Handle built-in commands.
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  exit - Exit the application")
                    print("  clear - Clear conversation history")
                    print("  Any other input will be treated as a question to the system")
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared")
                # Treat any other non-empty input as a query.
                elif user_input.strip():
                    print("\nProcessing your question...")
                    response = self.process_query(user_input)
                    print(f"\nResponse: {response}")
            # Allow the user to exit cleanly with Ctrl+C.
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    """Main entry point for the CLI."""
    cli = RagCLI()
    cli.run()

# Standard Python entry point check.
if __name__ == "__main__":
    main()