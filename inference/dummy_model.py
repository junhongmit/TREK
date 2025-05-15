from datetime import datetime


class DummyModel:
    def __init__(self, **kwargs):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        self.name = "dummy"
        pass

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

        return "i don't know"
