from datetime import datetime

from inference import *
from kg.kg_driver import *
from kg.kg_rep import *
from utils.prompt_list import *
from utils.utils import *
from utils.logger import *

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

PROMPTS["one_hop_kg_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are provided with a question in the {domain} domainm, its query time, and various references. Your task is to answer the question succinctly, using the fewest words possible.
        If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        """),

    "user": textwrap.dedent(
        """\
        ### References
        # Knowledge Graph
        {kg_results}
        ------
        Using only the references listed above, answer the following question: 
        Current Time: {query_time}
        Question: {query}
        """
    )
}


class OneHopKG_Model:
    """
    A one-hop KG based baseline from the CRAG benchmark.
    """

    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "one-hop-kg"
        self.domain = domain
        self.logger = logger

    @llm_retry(max_retries=10, default_output=[])
    async def extract_entity(
        self,
        query: str,
        query_time: datetime
    ) -> List[str]:
        system_prompt = PROMPTS["kg_topic_entity"]["system"]
        user_message = PROMPTS["kg_topic_entity"]["user"].format(
            query=query,
            query_time=query_time
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

        entity = maybe_load_json(response)
        return entity

    def get_kg_results(
        self,
        entity
    ):
        kg_results = []

        if isinstance(entity, dict) and "domain" in entity:
            domain = entity["domain"]

            # Handle Movie Queries
            if domain == "movie":
                results = []
                # Movie Information Queries
                if entity.get("movie_name"):
                    if isinstance(entity["movie_name"], str):
                        movie_names = entity["movie_name"].split(",")
                    else:
                        movie_names = entity["movie_name"]
                    for movie_name in movie_names:
                        results.extend(kg_driver.get_relations(source=KGEntity(
                            id="", type="Movie", name=normalize_entity(movie_name))))

                # Person Information Queries
                if entity.get("person"):
                    if isinstance(entity["person"], str):
                        person_list = entity["person"].split(",")
                    else:
                        person_list = entity["person"]
                    for person in person_list:
                        results.extend(kg_driver.get_relations(source=KGEntity(
                            id="", type="Person", name=normalize_entity(person))))

                # Movies Released in a Specific Year
                if entity.get("year"):
                    if isinstance(entity["year"], str) or isinstance(entity["year"], int):
                        years = str(entity["year"]).split(",")
                    else:
                        years = entity["year"]
                    for year in years:
                        results.extend(kg_driver.get_relations(source=KGEntity(
                            id="", type="Year", name=normalize_entity(str(year)))))

                kg_results = [relation_to_text(relation)
                              for relation in results]

            elif domain == "sports":
                results = []

                # Movie Information Queries
                if entity.get("tournament"):
                    if isinstance(entity["tournament"], str):
                        matches = entity["tournament"].split(",")
                    else:
                        matches = entity["tournament"]
                    for match in matches:
                        results = kg_driver.get_relations(source=KGEntity(
                            id="", type="Match", name=normalize_entity(match)))

                # Team Information Queries
                if entity.get("team"):
                    if isinstance(entity["team"], str):
                        teams = entity["team"].split(",")
                    else:
                        teams = entity["team"]
                    for team in teams:
                        results = kg_driver.get_relations(source=KGEntity(
                            id="", type="Team", name=normalize_entity(team)))

                kg_results = [relation_to_text(relation)
                              for relation in results]

            elif domain == "other":
                if entity.get("main_entity"):
                    if isinstance(entity["main_entity"], str):
                        entities = entity["main_entity"].split(",")
                    else:
                        entities = entity["main_entity"]
                    for entity in entities:
                        results = kg_driver.get_relations(source=KGEntity(
                            id="", type="", name=normalize_entity(entity)))

                kg_results = [relation_to_text(relation)
                              for relation in results]

        return "<DOC>\n".join([str(res) for res in kg_results]) if len(kg_results) > 0 else ""

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs
    ) -> str:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            query (str): User queries.
            query_time (datetime, optional): Timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            str: Plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        # Retrieve knowledge graph results
        entity = await self.extract_entity(query, query_time)

        kg_results = self.get_kg_results(entity)
        kg_results = kg_results[: MAX_CONTEXT_REFERENCES_LENGTH]

        # Prepare formatted prompts from the LLM
        system_prompt = PROMPTS["one_hop_kg_prompt"]["system"].format(
            domain=self.domain
        )
        user_message = PROMPTS["one_hop_kg_prompt"]["user"].format(
            kg_results=kg_results,
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
