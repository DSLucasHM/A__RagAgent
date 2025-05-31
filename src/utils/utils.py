import atexit
import pickle
import cachetools
import hashlib
import json
from typing import List

# === Relevance Cache with TTL and Hash ===

# An in-memory cache with a Time-To-Live (TTL) of 1 hour (3600 seconds).
# It stores up to 10,000 relevance judgements to avoid re-querying the LLM.
relevance_cache = cachetools.TTLCache(maxsize=10_000, ttl=3600)


def save_cache():
    """Saves the relevance cache to a pickle file."""
    # This function is registered to run automatically on program exit.
    with open('cache.pkl', 'wb') as f:
        # The cache object is converted to a dict for pickling.
        pickle.dump(dict(relevance_cache), f)


def load_cache():
    """Loads the relevance cache from a pickle file if it exists."""
    try:
        with open('cache.pkl', 'rb') as f:
            # Load the pickled dictionary and update the cache instance.
            relevance_cache.update(pickle.load(f))
    except (FileNotFoundError, EOFError, pickle.PickleError):
        # If the cache file doesn't exist or is corrupt, simply start fresh.
        pass


# Load the cache from disk when the module is first imported.
load_cache()
# Register the save_cache function to be called when the Python interpreter exits.
atexit.register(save_cache)


def _make_cache_key(query: str, doc_content: str) -> str:
    """Creates a unique cache key using a hash of the document and the query."""
    # Hashing the document content is more efficient than using the full text
    # and ensures a consistent key format.
    doc_hash = hashlib.sha256(doc_content.encode("utf-8")).hexdigest()
    return f"{doc_hash}:{query}"


def _parse_crag_output_json(
    raw_output: str, expected_batch_size: int
) -> List[dict]:
    """Parses the JSON output from the LLM robustly."""
    results = []
    try:
        # The LLM might wrap its JSON in markdown code fences; this removes them.
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        parsed_data = json.loads(raw_output)
        # The LLM might return a single object instead of a list for a single-item batch.
        if isinstance(parsed_data, list):
            results.extend(parsed_data)
        else:
            results.append(parsed_data)
    except (json.JSONDecodeError, IndexError) as e:
        # Catches errors if the LLM output is not valid JSON.
        print(f"❌ Error parsing JSON: {e}\nLLM Output:\n{raw_output}")

    # Define a default result to use when parsing fails or results are missing.
    default_error_result = {
        "score": 1.0,
        "justification": "LLM response error or parsing failed.",
        "confidence": "low"
    }
    # Ensure the function always returns a list of the expected size.
    while len(results) < expected_batch_size:
        results.append(default_error_result)
    return results[:expected_batch_size]


def crag_relevance_judge_batch(
    query: str, fragments: List[str]
) -> List[dict]:
    """
    Sends fragments in batches to the LLM, using an explicit index in the JSON
    to ensure correct mapping.
    """
    # Local import to avoid circular dependency issues.
    from ..config import llm

    # Define a character limit for each batch sent to the LLM to stay
    # within its context window.
    BATCH_CHAR_LIMIT = 4000
    # A dictionary to store results mapped to their original index.
    all_results_mapped = {}
    # A list of tuples (original_index, fragment_content) to process.
    remaining_fragments = list(enumerate(fragments))

    # Loop until all fragments have been processed.
    while remaining_fragments:
        llm_batch_content = []
        # This map links the index within the temporary prompt (e.g., 0, 1, 2)
        # back to the fragment's original index in the input list.
        llm_batch_prompt_map = {}
        current_batch_char_count = 0
        next_iteration_fragments = []

        # Build a single batch of fragments to send to the LLM.
        for original_idx, frag_content in remaining_fragments:
            # First, check if the result for this fragment is already cached.
            cache_key = _make_cache_key(query, frag_content)
            if cache_key in relevance_cache:
                all_results_mapped[original_idx] = relevance_cache[cache_key]
                continue  # Skip adding this fragment to the current LLM batch.

            # Add fragment to the batch if it fits within the character limit.
            if (current_batch_char_count + len(frag_content) <= BATCH_CHAR_LIMIT
                    or not llm_batch_content):  # Always add at least one item.
                prompt_idx = len(llm_batch_content)
                llm_batch_content.append(frag_content)
                llm_batch_prompt_map[prompt_idx] = original_idx
                current_batch_char_count += len(frag_content)
            else:
                # If it doesn't fit, save it for the next batch.
                next_iteration_fragments.append((original_idx, frag_content))

        # The fragments for the next loop are the ones that didn't fit.
        remaining_fragments = next_iteration_fragments

        # If the batch has content (i.e., not everything was cached), call the LLM.
        if llm_batch_content:
            # Construct the prompt with clear instructions for the LLM.
            # Asking for `prompt_fragment_index` is key for reliable mapping.
            prompt = f"""You are a relevance judge. For each document fragment below, evaluate its relevance to the user's query: "{query}"

Respond ONLY with a valid JSON list of objects. Each object MUST have these keys:
- "prompt_fragment_index": The 0-based index of the fragment AS IT APPEARS IN THIS PROMPT (e.g., 0, 1, 2,...).
- "score": A float from 1.0 to 10.0.
- "justification": A brief string explaining the score.
- "confidence": A string ("low", "medium", or "high").
Do not include any text or explanation before or after the JSON list.

Fragments:
"""
            for i, frag_c in enumerate(llm_batch_content):
                prompt += f"\n--- Fragment {i} ---\n{frag_c}"

            # Invoke the LLM and parse the output robustly.
            raw_output = llm.invoke(prompt).content
            parsed_results = _parse_crag_output_json(
                raw_output, len(llm_batch_content)
            )

            processed_indices_in_batch = set()
            # Map results back to their original indices using the explicit index.
            for res in parsed_results:
                prompt_idx = res.get("prompt_fragment_index")
                if (prompt_idx is not None and isinstance(prompt_idx, int) and
                        prompt_idx in llm_batch_prompt_map):
                    original_idx = llm_batch_prompt_map[prompt_idx]
                    # Create the dictionary to store, excluding the temporary index.
                    storage_dict = {
                        k: v for k, v in res.items()
                        if k != "prompt_fragment_index"
                    }
                    if original_idx not in all_results_mapped:
                        all_results_mapped[original_idx] = storage_dict
                        # Store the result in the cache for future use.
                        relevance_cache[
                            _make_cache_key(query, fragments[original_idx])
                        ] = storage_dict
                        processed_indices_in_batch.add(prompt_idx)

            # Fallback for results where the LLM failed to provide a valid index.
            # This attempts to map them based on their order.
            if len(processed_indices_in_batch) < len(parsed_results):
                unmapped_results = [
                    res for i, res in enumerate(parsed_results)
                    if i not in processed_indices_in_batch
                ]
                unmapped_prompt_indices = [
                    i for i in range(len(llm_batch_content))
                    if i not in processed_indices_in_batch
                ]
                for res, prompt_idx in zip(unmapped_results, unmapped_prompt_indices):
                    original_idx = llm_batch_prompt_map.get(prompt_idx)
                    if original_idx and original_idx not in all_results_mapped:
                        storage_dict = {
                            k: v for k, v in res.items()
                            if k != "prompt_fragment_index"
                        }
                        all_results_mapped[original_idx] = storage_dict
                        relevance_cache[
                           _make_cache_key(query, fragments[original_idx])
                        ] = storage_dict

    # Final assembly of results to ensure the output list is in the same
    # order as the input `fragments` list.
    final_results = []
    error_result = {
        "score": 1.0,
        "justification": "Result missing after all processing.",
        "confidence": "low"
    }
    for i in range(len(fragments)):
        final_results.append(all_results_mapped.get(i, error_result))
    return final_results