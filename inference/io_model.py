from datetime import datetime

from inference import *
from utils.prompt_list import *
from utils.utils import *

PROMPTS = get_default_prompts()

PROMPTS["io_prompt"] = {
    "system": textwrap.dedent(
        """\
        You are provided with a question in the {domain} domain, and its query time. Your task is to answer the question succinctly, using the fewest words possible. 
        If you don't have enough knowledge to answer the question, respond with 'I don't know'.
        """),

    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        """
    )
}


class IO_Model:
    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "io"
        self.domain = domain
        self.logger = logger

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs
    ) -> str:
        """
        Generates answers for a query using associated (pre-cached) search results and query times.

        Parameters:
            query (str): User queries.
            query_time (str): timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            str: A plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        system_prompt = PROMPTS["io_prompt"]["system"].format(
            domain=self.domain
        )
        user_message = PROMPTS["io_prompt"]["user"].format(
            query=query,
            query_time=query_time
        )

        response = await generate_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
            max_tokens=75)
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

        return response
