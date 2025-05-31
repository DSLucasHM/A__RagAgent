from typing import List
import glob
import os
import fitz
from langchain_core.documents import Document as LC_Document
from langchain_community.document_loaders import WebBaseLoader
from ..config import search_tool
from ..utils.utils import crag_relevance_judge_batch
import bert_score
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from ..config import bert_model


class VectorDB:
    """A simple in-memory vector database."""
    def __init__(self):
        """Initializes the vector database with empty lists."""
        # Stores the original LangChain Document objects.
        self.documents = []
        # Stores the sentence-transformer embeddings as a single PyTorch tensor.
        self.embeddings = []

    def add_documents(self, docs: List[LC_Document]):
        """Adds documents and their embeddings to the database."""
        if not docs:
            # Do nothing if the input list is empty.
            return
        # Extend the list of document objects.
        self.documents.extend(docs)
        # Encode the text content of the new documents into vector embeddings.
        new_embeddings = bert_model.encode(
            [doc.page_content for doc in docs], convert_to_tensor=True
        )
        # Concatenate the new embeddings with any existing ones.
        if len(self.embeddings) > 0:
            self.embeddings = util.cat((self.embeddings, new_embeddings))
        else:
            # If this is the first batch, initialize the embeddings tensor.
            self.embeddings = new_embeddings


class HybridRetriever:
    """
    A retriever that combines lexical and semantic search with MMR for diversity.
    """
    def __init__(self, docs: List[LC_Document], scorer):
        """Initializes the retriever and pre-computes necessary components."""
        # The original documents to be searched.
        self.docs = docs
        # The text content extracted from the documents.
        self.texts = [doc.page_content for doc in docs]
        # Pre-tokenize the texts for the BM25 lexical search algorithm.
        self.tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)
        # Pre-compute embeddings for all texts for semantic search.
        self.embeddings = bert_model.encode(self.texts, convert_to_tensor=True)
        # The BERTScorer instance for semantic similarity scoring.
        self.scorer = scorer

    def retrieve(
        self,
        query: str,
        k: int = 5,
        alpha_bm25=0.3,
        alpha_cos=0.3,
        alpha_bert=0.4,
        mmr_lambda=0.6
    ) -> List[LC_Document]:
        """Retrieves k docs using a hybrid score and MMR for re-ranking."""
        # --- 1. Score Calculation ---
        # Calculate lexical scores using BM25.
        query_tok = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tok)

        # Calculate semantic scores using Cosine Similarity.
        query_emb = bert_model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]

        # Calculate semantic scores using BERTScore's F1 measure.
        # This provides a different kind of semantic relevance signal.
        _, _, f1 = self.scorer.score(
            [query] * len(self.texts), self.texts
        )
        f1_scores = f1.tolist()

        # --- 2. Hybrid Score Combination ---
        # Combine the three scores into a single hybrid score using weighted alphas.
        hybrid_scores = [
            (alpha_bm25 * bm25_scores[i]) +
            (alpha_cos * cos_scores[i].item()) +
            (alpha_bert * f1_scores[i])
            for i in range(len(self.texts))
        ]

        # --- 3. MMR Re-ranking for Diversity ---
        # Maximal Marginal Relevance (MMR) selects documents that are both
        # relevant to the query and diverse from each other.
        selected_indices = []
        if not hybrid_scores:
            return []

        # Start with the most relevant document based on the hybrid score.
        best_idx = max(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i])
        selected_indices.append(best_idx)
        # All other documents are initial candidates.
        candidate_indices = list(set(range(len(self.texts))) - set(selected_indices))

        # Iteratively add documents to the selection until k documents are chosen.
        while len(selected_indices) < k and candidate_indices:
            best_mmr_score = -float("inf")
            best_next_idx = None
            # Iterate through remaining candidates to find the next best one.
            for i in candidate_indices:
                # Get the candidate's relevance to the query.
                relevance = hybrid_scores[i]
                selected_embeddings = self.embeddings[selected_indices]
                # Calculate the candidate's similarity to already selected documents.
                diversity_scores = util.pytorch_cos_sim(self.embeddings[i], selected_embeddings)
                max_similarity = max(diversity_scores[0]).item()
                # The MMR score balances relevance against maximum similarity (redundancy).
                mmr_score = (mmr_lambda * relevance) - ((1 - mmr_lambda) * max_similarity)
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_next_idx = i

            # Add the best candidate to the selected list and remove it from candidates.
            if best_next_idx is not None:
                selected_indices.append(best_next_idx)
                candidate_indices.remove(best_next_idx)
            else:
                # Stop if no suitable candidates are left.
                break
        # Return the final list of selected documents.
        return [self.docs[i] for i in selected_indices]


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move the start position back by the overlap amount for the next chunk.
        start += chunk_size - chunk_overlap
    return chunks


def chunk_documents(docs: List[LC_Document]) -> List[LC_Document]:
    """Chunks a list of documents."""
    chunked_docs = []
    for doc in docs:
        # Split the page content of each document into text chunks.
        chunks = chunk_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            # Create a new Document object for each chunk, preserving the
            # original metadata and adding the chunk number.
            chunked_docs.append(LC_Document(
                page_content=chunk, metadata={**doc.metadata, "chunk": i}
            ))
    return chunked_docs


def load_documents_from_directory(directory: str) -> List[LC_Document]:
    """Loads supported text and PDF files from a directory."""
    docs = []
    # Search for .txt and .md files recursively.
    for ext in ["**/*.txt", "**/*.md"]:
        for path in glob.glob(os.path.join(directory, ext), recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append(LC_Document(page_content=content, metadata={"source": path}))
            except Exception as e:
                print(f"Error loading {path}: {e}")
    # Search for .pdf files recursively.
    for path in glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True):
        try:
            # Use PyMuPDF (fitz) to open and extract text from the PDF.
            doc = fitz.open(path)
            content = "\n".join([page.get_text() for page in doc])
            docs.append(LC_Document(page_content=content, metadata={"source": path}))
        except Exception as e:
            print(f"Failed to load PDF {path}: {e}")
    return docs


def process_urls(urls: List[str]) -> List[LC_Document]:
    """Loads and processes documents from a list of URLs."""
    # Use LangChain's WebBaseLoader to fetch and parse web content.
    loader = WebBaseLoader(urls)
    return loader.load()


def process_topics(topics: List[str]) -> List[LC_Document]:
    """Searches for topics and creates documents from the results."""
    docs = []
    for topic in topics:
        # Use the configured search tool (e.g., Tavily) to get results.
        results = search_tool.invoke(topic)
        for res in results:
            # Create a Document object from the content of each search result.
            if content := res.get("content"):
                docs.append(LC_Document(
                    page_content=content,
                    metadata={"source": res.get("url"), "topic": topic}
                ))
    return docs