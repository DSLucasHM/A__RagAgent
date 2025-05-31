# RAG Stock Research Agent

---

### 👨‍💻 Author

---

**Lucas Miyazawa**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lucasmiyazawa/) [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:lucasmiyazawa@icloud.com)

### 📚 About the Project

---


📚 About the Project
This AI-powered research system uses an advanced agentic workflow to deliver accurate, relevant, and diverse financial information. The agent intelligently routes user queries, deciding whether to engage in simple conversation or to launch a full research pipeline. When research is needed, it uses a hybrid search algorithm and Corrective-RAG (CRAG) techniques to ensure the information used for generating answers is of the highest quality and relevance.

Technically, the system leverages LangGraph to orchestrate the multi-step agent workflow. It combines multiple search signals (BM25, Cosine Similarity, BERTScore) and re-ranks results with Maximal Marginal Relevance (MMR) for diversity. OpenAI and Groq models power the reasoning and generation, while Tavily provides real-time web search capabilities. The architecture is modular, scalable, and designed for a sophisticated, interactive command-line experience.

🗂️ Project Structure
Plaintext

stock-research-agent/
├── src/
│   ├── core/
│   │   ├── agent/
│   │   │   └── graph.py        # Execution graph definition
│   │   ├── nodes.py            # Implementation of graph nodes
│   │   └── state.py            # State management for the graph
│   ├── retriever/
│   │   └── search.py           # HybridRetriever and VectorDB logic
│   ├── utils/
│   │   └── utils.py            # Helper functions (e.g., CRAG judge)
│   ├── config.py               # Centralized configurations and models
│   └── main.py                 # Application entry point (CLI)
├── documents/                  # Folder for local source documents
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # Project documentation

🛠️ Agent Features
Dynamic Query Routing: Intelligently distinguishes between conversational queries and deep research questions.
Hybrid Search: Combines keyword (BM25) and semantic search for robust retrieval.
CRAG Validation: Uses an LLM to score and filter retrieved documents for relevance, reducing noise.
MMR Re-Ranking: Ensures final context is both relevant and diverse, avoiding redundant information.
Interactive CLI Session: Provides a stateful command-line interface for continuous conversation.
Persistent Caching: Caches models and relevance scores to accelerate subsequent runs.


🔄 Clone the Repository
Clone this repository into your machine:

```bash
git clone https://github.com/your-username/rag_agent.git
```

📂 Navigate to the Project Folder
Move into the cloned project directory:

```bash
cd rag_agent
```


This RAG application builds its knowledge from the data sources you provide. You have three ways to feed information into the system: using web URLs, search topics, or local files.

### 1. Using the `.env` File (for URLs and Topics)

The primary way to configure the data sources is by editing the `.env` file in the root of the project.

-   **`INGEST_URLS`**: Use this variable to provide one or more web page URLs that you want the application to read and learn from.
    -   **Format**: Separate multiple URLs with a comma (`,`).
    -   **Example**:
        ```
        INGEST_URLS=[https://www.theverge.com/ai-artificial-intelligence,https://www.reuters.com/technology/artificial-intelligence/](https://www.theverge.com/ai-artificial-intelligence,https://www.reuters.com/technology/artificial-intelligence/)
        ```

-   **`INGEST_TOPICS`**: Use this variable to provide search topics. The application will use a search tool to find relevant, up-to-date information on these topics.
    -   **Format**: Separate multiple topics with a comma (`,`).
    -   **Example**:
        ```
        INGEST_TOPICS=latest AI hardware,AI impact on creative jobs
        ```

### 2. Using the `documents/` Directory (for Local Files)

To process local files (such as `.pdf`, `.txt`, `.md`, etc.), simply place them inside the `documents/` folder located at the root of this project.

The application will automatically find and load any supported files it finds in this directory when it starts up.

### ✅ Try It Now: Sample Data Included!

To help you get started immediately, **this project already includes sample data for all three methods!**

-   **Sample URLs**: The `.env` file is pre-configured with sample URLs for you to test.
-   **Sample Topics**: The `.env` file also contains sample search topics.
-   **Sample Document**: There is a sample document (`sample_document.txt`) already placed inside the `documents/` directory.

This means you can **run the application right away without changing anything** to see how it works!


```

⚙️ Set Up Environment Variables
Create a .env file from the provided example:

```bash
cp .env.example .env
```

🔑 Add Your API Keys
Open the .env file and set your API keys:

```
OPENAI_API_KEY="your_openai_key"       # https://platform.openai.com/api-keys
GROQ_API_KEY="your_groq_key"           # https://console.groq.com/keys
TAVILY_API_KEY="your_tavily_key"       # https://app.tavily.com/home
LANGSMITH_API_KEY="your_langsmith_key" # https://smith.langchain.com/settings
```

🐳 How to Run with Docker (Recommended)
The recommended workflow separates the one-time data ingestion from the daily interactive use.

Build the Docker Image:
In the project's root directory, run:

```bash
docker build -t rag_agent .
```

Run Data Ingestion (One-Time Setup):
This command runs the container in a special "ingest mode" to process your source documents and create a persistent vector database. This should only be done once, or when you update your source documents. (This requires the main.py to be modified to handle the APP_MODE environment variable.)

```bash
docker run -it  --rm --name rag_agent -p 5000:5000 --env-file .rag_agent
```

⚠️ Note: To exit the interactive session, you can type exit in the CLI or press Ctrl+C.

💻 How to Run Locally for Development (Without Docker)

Create and Activate a Virtual Environment:
In the root directory:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

Run the CLI Application:
From the root directory:

```bash
python -m src.main
```

⚠️ Considerations
- API Keys: The application will not function without the required API keys set in the .env file.
- Local Execution: When running locally without Docker, the vector database will be rebuilt from scratch on every run, which can be slow. The Docker workflow is recommended to avoid this.

🔮 Future Implementations
- Develop a simple web interface (Flask/FastAPI) as an alternative to the CLI.
- Integrate more data sources, such as financial data APIs or SEC filings.
- Add support for generating and saving analysis reports to different formats (PDF, DOCX).
- Implement more advanced financial analysis tools and agent capabilities.

🔗 References
- https://langchain.com/
- https://openai.com
- https://groq.com/
- https://tavily.com
