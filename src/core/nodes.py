
from .state import  ConversationState
from ..utils.utils import crag_relevance_judge_batch
from ..retriever.search import HybridRetriever
from ..config import slm,llm, bert_scorer



# Note: The `llm`, `slm`, `bert_scorer`, and `HybridRetriever` are assumed
# to be imported or available in the scope where these functions are used,
# typically managed by the application's entry point or graph definition.



def transform_question(state: ConversationState) -> dict:
    """Transforms the user's question for better retrieval."""
    print("▶️  Entered node: transform_question")
    question = state.current_question

    # For very short or simple queries, skip the transformation step
    # to save time and avoid altering simple questions.
    if len(question.split()) < 5:
        return {"transformed_question": question}

    # This prompt instructs an LLM to rephrase the user's conversational
    # question into a concise, keyword-focused query suitable for searching.
    prompt = f"""Transform the following user question into a search-optimized query that will help retrieve relevant documents:

User Question: {question}

Your task:
1. Identify the core information need
2. Remove filler words and conversational elements
3. Extract key terms and concepts
4. Format as a concise, search-friendly query
5. Preserve all important entities and relationships

Transformed Query:"""

    # Invoke the language model to perform the transformation.
    transformed = llm.invoke(prompt).content.strip()
    print(f"🔄 Transformed: '{question}' → '{transformed}'")
    return {"transformed_question": transformed}


def validate_documents_with_crag(state: ConversationState) -> dict:
    """Validates retrieved documents using the batched CRAG approach."""
    print("▶️  Entered node: validate_documents_with_crag")
    # If no documents were retrieved, there's nothing to validate.
    if not state.retrieved_docs:
        return {"retrieved_docs": []}

    # Use the transformed query if available, otherwise use the original.
    query = state.transformed_question or state.current_question
    fragments = [doc.page_content for doc in state.retrieved_docs]

    # Call a utility function to have an LLM judge the relevance of each fragment.
    relevance_results = crag_relevance_judge_batch(query, fragments)

    # Filter the documents, keeping only those that meet the relevance threshold.
    # This is the "Corrective" part of Corrective-RAG.
    filtered_docs = []
    for i, (doc, result) in enumerate(zip(state.retrieved_docs, relevance_results)):
        score = result.get("score", 0)
        if score >= 5.0:  # Threshold for relevance
            filtered_docs.append(doc)
            print(f"✅ Doc {i}: Score {score:.1f} - {result.get('justification', 'No justification')[:50]}...")
        else:
            print(f"❌ Doc {i}: Score {score:.1f} - {result.get('justification', 'No justification')[:50]}...")

    print(f"🔍 Filtered {len(filtered_docs)}/{len(state.retrieved_docs)} documents.")
    return {"retrieved_docs": filtered_docs}


def generate_response(state: ConversationState) -> dict:
    """Generates a response based on the retrieved documents."""
    print("▶️  Entered node: generate_response")

    # Format the content from the validated, retrieved documents into a single string.
    doc_contents = []
    for i, doc in enumerate(state.retrieved_docs):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content
        doc_contents.append(f"Document {i+1} [Source: {source}]:\n{content}")

    # Create the final context string for the prompt.
    context = "\n\n".join(doc_contents) if doc_contents else "No relevant documents found."

    # Build a string from the last few turns of conversation history for context.
    conversation_context = ""
    if state.conversation_history:
        for i, exchange in enumerate(state.conversation_history[-3:]):  # Last 3 exchanges
            if "user" in exchange:
                conversation_context += f"User: {exchange['user']}\n"
            if "assistant" in exchange:
                conversation_context += f"Assistant: {exchange['assistant']}\n"

    # Construct the final prompt for the LLM, including all available context.
    prompt = f"""{state.system_prompt}

Conversation History:
{conversation_context}
User Question: {state.current_question}

Context from retrieved documents:
{context}

Instructions:
- Answer the user's question based on the provided context
- If the context doesn't contain relevant information, say so honestly
- Cite specific documents when referencing information
- Be conversational and helpful

Your response:"""

    # Invoke the main LLM to generate the final answer.
    response = llm.invoke(prompt).content.strip()

    # Update the conversation history with the latest user query and AI response.
    conversation_history = state.conversation_history.copy()
    conversation_history.append({
        "user": state.current_question,
        "assistant": response
    })

    return {
        "response": response,
        "conversation_history": conversation_history
    }


def retrieve_documents_node(state: ConversationState, vector_db) -> dict:
    """
    Searches documents in a provided vector database.
    This is a pure function; its only external dependency is the `vector_db` passed as an argument.
    """
    print("▶️  Entered node: retrieve_documents_node (from nodes.py)")
    # Use the transformed query for retrieval if it exists.
    query = state.transformed_question or state.current_question

    # Initialize the hybrid retriever with the documents from the VectorDB.
    # The `vector_db` object is passed in via `functools.partial` in the main script.
    retriever = HybridRetriever(docs=vector_db.documents, scorer=bert_scorer)
    retrieved = retriever.retrieve(query)

    return {"retrieved_docs": retrieved}


def route_question(state: ConversationState) -> dict:
    """
    Routes the user's question to decide if it needs RAG or can be answered directly.
    """
    print("▶️  Entered node: route_question")
    question = state.current_question

    # This prompt asks a fast model to classify the user's intent.
    prompt = f"""You are an expert at routing a user question.
Analyze the user's question and determine if it's a simple, conversational query (like a greeting, a thank you, or a farewell) or if it requires looking up specific information from a knowledge base (RAG).

The user question is: "{question}"

Possible routes are:
1. 'simple_response': For greetings, thank yous, farewells, and other general conversational queries.
2. 'rag_pipeline': For any question that asks for specific information, details, or explanations.

Return a single JSON object with a 'decision' key. For example:
{{"decision": "simple_response"}}

Do not return anything more than
{{"decision": "simple_response"}}
"""

    # Use the smaller, faster model (slm) for this routing task to improve latency.
    response = slm.invoke(prompt)

    try:
        # Attempt to parse the JSON response from the model.
        import json
        decision_data = json.loads(response.content)
        decision = decision_data.get("decision", "rag_pipeline")
        print(f"🚦 Decision: The route is '{decision}'")
        return {"route_decision": decision}
    except Exception:
        # If the model's output is not valid JSON or another error occurs,
        # default to the safer, more comprehensive RAG pipeline.
        print("🚦 Decision: Parsing failed, defaulting to 'rag_pipeline'")
        return {"route_decision": "rag_pipeline"}


def generate_simple_response(state: ConversationState) -> dict:
    """
    Generates a direct, simple response for conversational queries.
    """
    print("▶️  Entered node: generate_simple_response")
    question = state.current_question

    # This prompt is for general conversation and does not use retrieved context.
    prompt = f"""You are a friendly conversational assistant. Respond directly to the user's message in a helpful and concise way.

User's message: "{question}"

Your response:"""

    # Use the main LLM to ensure a high-quality conversational response.
    response = llm.invoke(prompt).content.strip()

    # Update the conversation history, same as in the main response generator.
    conversation_history = state.conversation_history.copy()
    conversation_history.append({
        "user": state.current_question,
        "assistant": response
    })

    return {
        "response": response,
        "conversation_history": conversation_history
    }