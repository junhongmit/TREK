from datetime import datetime
from blingfire import text_to_sentences_and_offsets
from collections import defaultdict
import numpy as np
import ray
from sentence_transformers import SentenceTransformer
import torch
from typing import Any, Dict, List

from inference import *
from utils.prompt_list import *
from utils.utils import *

######################################################################################################
######################################################################################################
###
# Please pay special attention to the comments that start with "TUNE THIS VARIABLE"
# as they depend on your model and the available GPU resources.
###
# DISCLAIMER: This baseline has NOT been tuned for performance
# or efficiency, and is provided as is for demonstration.
######################################################################################################

# CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# CONFIG PARAMETERS END---

PROMPTS = get_default_prompts()

PROMPTS["rag_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are provided with a question in the {domain} domain, its query time, and various references. Your task is to answer the question succinctly, using the fewest words possible. 
        If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers.
        """),

    "user": textwrap.dedent(
        """\
        {references}
        ------

        Using only the references listed above, answer the following question:
        Question: {query}
        Query Time: {query_time}
        """
    )
}


class ChunkExtractor:

    @ray.remote
    def _extract_chunks(
        self,
        interaction_id,
        doc
    ):
        """
        Extracts and returns chunks from given doc.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this doc belongs to.
            doc (str): doc content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the doc content.
        """
        if not doc:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(doc)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = doc[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunk(
        self,
        interaction_id,
        docs
    ):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            interaction_ids (str): interaction ID.
            docs (List[Dict]): List of search results, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=interaction_id,
                doc=doc
            )
            for doc in docs
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            # Blocking call until parallel execution is complete
            interaction_id, _chunks = ray.get(response_ref)
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def extract_chunks(
        self,
        batch_interaction_ids,
        batch_docs
    ):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_docs (List[List[str]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                doc=doc
            )
            for idx, docs in enumerate(batch_docs)
            for doc in docs
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            # Blocking call until parallel execution is complete
            interaction_id, _chunks = ray.get(response_ref)
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(
        self,
        chunk_dictionary
    ):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class RAG_Model:
    """
    An example Retrieval-Augmented-Generation (RAG) Model
    """

    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "rag"
        self.domain = domain
        self.logger = logger

        self.chunk_extractor = ChunkExtractor()

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        interaction_id: str = None,
        docs: List[str] = None,
        **kwargs
    ) -> str:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'docs' (List[List[Dict]]): List of doc lists, each corresponding
                                                      to a query.
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        # Chunk all search results using ChunkExtractor
        chunks, _ = self.chunk_extractor.extract_chunk(
            interaction_id, docs
        )

        # Calculate all chunk embeddings
        chunk_embeddings = np.array(await generate_embedding(chunks))

        # Calculate embeddings for queries
        query_embedding = np.array(await generate_embedding([query]))[0]

        # Calculate cosine similarity between query and chunk embeddings,
        cosine_scores = (chunk_embeddings * query_embedding).sum(1)

        # and retrieve top-N results.
        retrieval_results = chunks[
            (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
        ]

        # Prepare formatted prompts from the LLM
        system_prompt = PROMPTS["rag_prompt"]["system"].format(
            domain=self.domain
        )
        references = ""
        if len(retrieval_results) > 0:
            references += "# References \n"
            # Format the top sentences as references in the model's prompt template.
            for _snippet_idx, snippet in enumerate(retrieval_results):
                references += f"- {snippet.strip()}\n"
        references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

        user_message = PROMPTS["rag_prompt"]["user"].format(
            references=references,
            query=query,
            query_time=query_time
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=75
        )
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

        return response
