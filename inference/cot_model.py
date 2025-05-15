from datetime import datetime

from inference import *
from utils.prompt_list import *
from utils.utils import *

PROMPTS = get_default_prompts()

PROMPTS["cot_prompt"] = {
    "system": textwrap.dedent(
        """\
        -Goal-
        You are provided with a question in the {domain} domain, and its query time. Your task is to answer the question succinctly, using the fewest words possible. 
        If you don't have enough knowledge to answer the question, respond with 'I don't know'.

        Let's think step by step, and return your judgment in a JSON of the format {{"reason": "...", "answer": "..."}} (TIP: You will need to escape any double quotes in the string to make the JSON valid).
        
        #### Examples ####
        Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
        Output: {{"reason": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is Washington, D.C.",
                "answer": "Washington, D.C."}}

        Question: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
        Output: {{"reason": "First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is Bharoto Bhagyo Bidhata.",
                "answer": "Bharoto Bhagyo Bidhata"}}
                                

        Question: Who was the artist nominated for an award for You Drive Me Crazy?
        Output: {{"reason": "First, the song 'You Drive Me Crazy' was performed by Britney Spears. Second, Britney Spears was nominated for awards for this song. The answer is Britney Spears.",
                "answer": "Britney Spears"}}
                                

        Question: What person born in Siegen influenced the work of Vincent Van Gogh?
        Output: {{"reason": " First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is Peter Paul Rubens.",
                "answer": "Peter Paul Rubens"}}
                                

        Question: What is the country close to Russia where Mikheil Saakashvii holds a government position?
        Output: {{"reason": "First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is Georgia.",
                "answer": "Georgia"}}
                                

        Question: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
        Output: {{"reason": "First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is Heroin.",
                "answer": "Heroin"}}
        """
    ),

    "user": textwrap.dedent(
        """\
        Question: {query}
        Query Time: {query_time}
        Output:
        """
    )
}


class CoT_Model:
    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
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
            str: A plain text responses for each query in the batch.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        system_prompt = PROMPTS["cot_prompt"]["system"].format(
            domain=self.domain
        )
        user_message = PROMPTS["cot_prompt"]["user"].format(
            query=query,
            query_time=query_time
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

        return maybe_load_json(response)["answer"]
