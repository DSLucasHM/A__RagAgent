import os
from dotenv import load_dotenv


# === Models ===
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from bert_score import BERTScorer




# Loads environment variables from the .env file
load_dotenv()



# API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT")

# LangChain configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"



# Main Large Language Model (LLM) for high-quality generation tasks.
# Uses OpenAI's gpt-4o-mini model.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# A tool for performing web searches to augment the context.
search_tool = TavilySearchResults( =1)

# A sentence-transformer model used to create vector embeddings from text.
# These embeddings are essential for semantic search in the vector database.
bert_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# A smaller, faster language model (SLM) served via the Groq API.
# Used for simpler, quicker tasks like routing or generating prompts.
slm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)


# A scorer for evaluating semantic similarity between texts.
# Likely used for re-ranking retrieved documents or validating answers.
bert_scorer = BERTScorer(lang="en", model_type="xlm-roberta-large")




# Enables LangChain tracing for debugging and monitoring through LangSmith.
os.environ["LANGCHAIN_TRACING_V2"] = "true"




# Read the URL string and topics from the environment
urls_str = os.environ.get("INGEST_URLS", "")  
INGEST_URLS = [url.strip() for url in urls_str.split(',') if url.strip()]

topics_str = os.environ.get("INGEST_TOPICS", "")
INGEST_TOPICS = [topic.strip() for topic in topics_str.split(',') if topic.strip()]