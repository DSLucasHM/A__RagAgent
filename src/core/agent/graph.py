from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from ..state import ConversationState
from ..nodes import (
    route_question, 
    generate_simple_response,
    retrieve_documents_node, # This is the original node passed as an argument
    validate_documents_with_crag, 
    generate_response
                    )






def maybe_transform_query(state: ConversationState) -> dict:
    """Rewrites the query if no relevant documents were found."""
    # This node is a fallback mechanism. It's only called if retrieval and
    # validation result in an empty list of documents.
    if not state.retrieved_docs:
        print("🤔 No relevant documents. Transforming the query.")
        # Construct a prompt to have a fast LLM rewrite the original question.
        prompt = (f"The question '{state.current_question}' did not return "
                  f"relevant documents. Rewrite the question to improve results.")
        new_query = slm.invoke(prompt).content.strip()
        # Return the new query to be used in another retrieval attempt.
        return {"transformed_question": new_query, "retrieved_docs": []}
    # This branch is unlikely to be hit due to the graph's conditional logic,
    # which only routes to this node when retrieved_docs is empty.
    return {"transformed_question": None}

def decide_after_routing(state: ConversationState) -> str:
    """Decides the next step after the initial routing."""
    # This function acts as a conditional edge in the graph.
    # It checks the decision made by the 'router' node.
    if state.route_decision == "simple_response":
        # If the query is conversational, go to the simple response node.
        return "simple_response"
    else:
        # Otherwise, proceed with the full RAG pipeline.
        return "retrieve"

def decide_after_validation(state: ConversationState) -> str:
    """Decides the next step after validation."""
    # This conditional edge is called after the CRAG validation node.
    # If there are relevant documents remaining, proceed to answer generation.
    return "answer" if state.retrieved_docs else "maybe_transform"

def build_conversation_workflow(retrieve_node):
    """
    Builds the conversation workflow graph with the new routing logic.

    Args:
        retrieve_node: A pre-configured function that retrieves documents.

    Returns:
        A compiled StateGraph for the conversation workflow.
    """
    # Initialize the state machine graph with the central state object.
    workflow = StateGraph(ConversationState)

    # Add all the functions as nodes in the graph.
    workflow.add_node("router", route_question)
    workflow.add_node("simple_response", generate_simple_response)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("validate", validate_documents_with_crag)
    workflow.add_node("maybe_transform", maybe_transform_query)
    workflow.add_node("answer", generate_response)

    # --- Define the graph's edges and flow ---

    # 1. The workflow starts at the "router" node.
    workflow.set_entry_point("router")

    # 2. From the router, we conditionally branch based on the query type.
    workflow.add_conditional_edges(
        "router",
        decide_after_routing,
        {
            # Maps the string returned by the decider function to a node name.
            "simple_response": "simple_response",
            "retrieve": "retrieve"
        }
    )

    # 3. The "simple_response" path is a terminal path; it ends the graph run.
    workflow.add_edge("simple_response", END)

    # 4. Define the main RAG pipeline path with its own conditional logic.
    workflow.add_edge("retrieve", "validate")
    workflow.add_conditional_edges(
        "validate",
        decide_after_validation,
        {
            # If validation succeeds, go to the answer node.
            "answer": "answer",
            # If validation fails, attempt to rewrite the query.
            "maybe_transform": "maybe_transform"
        }
    )
    # This creates the retry loop: after transformation, retrieve again.
    workflow.add_edge("maybe_transform", "retrieve")
    # After generating the RAG answer, the graph run ends.
    workflow.add_edge("answer", END)

    # Compile the graph into a runnable object.
    # The checkpointer enables memory, so the graph can be stateful
    # across multiple invocations with the same thread_id.
    return workflow.compile(checkpointer=MemorySaver())