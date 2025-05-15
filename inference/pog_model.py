import asyncio
import json
import random
from typing import Any, Dict, List

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

PROMPTS = get_default_prompts()

PROMPTS["subobjective_prompt"] = """Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis, and return the output as a list of string.
Here is an example: 
Q: Which of the countries in the Caribbean has the smallest country calling code?
Output: ['Search the countries in the Caribbean', 'Search the country calling code for each Caribbean country', 'Compare the country calling codes to find the smallest one']

Now you need to directly output subobjectives of the following question in list format without other information or notes. 
Q: {query}"""

PROMPTS["relations_pruning"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are given a question in the {domain} domain, its query time, its subobjectives, an entity, and a list of relations starting from it.
    The goal is to retrieve up to {width} relations that contribute to answering the question and rate their relevance from 0 to 1 (use at most 3 decimal places, and remove trailing zeros; the sum of the scores of these relations is 1).

    -Steps-
    1. You are provided a list of directed relations between entities in the format of
    rel_i: (entity_type: entity_name)-[relation_type, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(entity_type: entity_name).
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.

    2. Retrieve relations only from the given list that contribute to answering the question, and provide a short reason for your scoring.
    Return its index (rel_i) and score into a json of the format: {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}.
    (TIP: You will need to escape any double quotes in the string to make the JSON valid)
    
    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
    
    ######################
    -Examples-
    Question: Which movie wins the best visual effect award in 2006 Oscars?
    Entity: (Award: VISUAL EFFECTS, properties: <year: 2006, ceremony_number: 78, type: OSCAR AWARD>)
    Relations: rel_0: (Award: VISUAL EFFECTS)-[HELD_IN]->(Year: None)
    rel_1: (Award: VISUAL EFFECTS)-[NOMINATED_FOR, properties: <winner, person, movie>]->(Movie: None)
    rel_2: (Award: VISUAL EFFECTS)-[WON, properties: <winner, person, movie>]->(Movie: None)
    Output: {{"reason": "The question is asking for movies that won the award, relation rel_2 is the most relevant to award winning. rel_1 is relation that find movies released in 2006\
    and may help find the movie that wins the award. A movie that won the award should also got nominated for the award, so rel_1 also has slight relevance. ", 
    "relevant_relations": {{"rel_2": 0.7, "rel_0": 0.2, "rel_1": 0.1}}
    }}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Subobjectives: {subqueries}
    Topic Entity: {entity}
    Relations: {relations}
    
    Output Format (flat JSON): {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}
    Output:
    """)
}

PROMPTS["triplets_pruning"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are given a question in the {domain} domain, its query time, a list of directed relations in the format of (source entity)-[relation]->(target entity).
    The goal is to score the relations' contribution to the question on a scale from 0 to 1 (use at most 3 decimal places, and remove trailing zeros; the sum of the scores of all relations is 1).
    
    -Steps-
    1. You are provided a list of directed relations in the format of
    rel_i: (entity_type: entity_name)-[relation_type, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(entity_type: entity_name, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
    You are going to assess the relevance of the relation type and its properties, along with the target entity name and its properties, to the given question.
    
    2. Score the relations' relevance to answering the question, and provide a short reason for your scoring.
    Return its index (ent_i) and score into a valid JSON of the format: {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}.
    (TIP: You will need to escape any double quotes in the string to make the JSON valid)
    
    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
    
    ######################
    -Examples-
    ######################
    Question: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
    Relations: rel_0: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: The Resident)
    rel_1: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: So Undercover, properties: <featured: Miley Cyrus, Jeremy Piven, and Mike O'Malley>)
    rel_2: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: Let Me In, properties: <featured: Kodi Smit-McPhee, Chloë Grace Moretz, Elias Koteas, Cara Buono, and Richard Jenkins>)
    rel_3: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: Begin Again, properties: <featured: Keira Knightley, Mark Ruffalo, Hailee Steinfeld>)
    rel_4: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: A Walk Among the Tombstones, properties: <featured: Liam Neeson, Dan Stevens, David Harbour>)
    Output: {{"reason": "The movie that matches the given criteria is 'So Undercover' with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for 'So Undercover' would be 1, and the scores for all other entities would be 0.", "relevant_relations": {{"rel_1": 1.0}}}}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Relations: {relations}
    
    Output Format (flat JSON): {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}
    Output:
    """)
}

PROMPTS["update_mem_prompt"] = """Based on the provided information (which may have missing parts and require further retrieval) and your own knowledge, output the currently known information required to achieve the subobjectives.
Here is an example:
Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Subobjectives: ['Search the person who said "Taste cannot be controlled by law"', 'Search the cause of death for that person']
Memory: 
Knowledge Triplets: Taste cannot be controlled by law. media_common.quotation.author [Thomas Jefferson]
Output: {{
    "1": "Thomas Jefferson said 'Taste cannot be controlled by law'.",
    "2": "It is not mentioned, and I also don't know."
}}

Now you need to directly output the results of the following question in JSON format without other information or notes. 
Q: {query}
Subobjectives: {subqueries}
Memory: {memory}
Knowledge Triplets: {relations}
Output:
"""

PROMPTS["answer_depth_prompt"] = """Please answer the question based on the memory, related knowledge triplets and your knowledge.

Here are five examples:
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Memory: {{
    "1": "The triplet provides the information that Thomas Jefferson said this sentence.",
    "2": "No triplet provides this information."
}}
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, [Thomas Jefferson]
Output:
{{
    "A": {{
        "Sufficient": "No",
        "Answer": "Null"
    }},
    "R": "The person who said "Taste cannot be controlled by law," is Thomas Jefferson. It is still uncertain to answer the entire question"
}}

Q: The artist nominated for The Long Winter lived where?
Memory: {{
    "1": "The triplets provide the information that the author of The Long Winter is Laura Ingalls Wilder.",
    "2": "The triplets provide this information that Laura Ingalls Wilder lived in De Smet."
}}
Knowledge Triplets: The Long Winter, book.written_work.author, [Laura Ingalls Wilder]
Laura Ingalls Wilder, people.person.places_lived, [Unknown-Entity]
Unknown-Entity, people.place_lived.location, [De Smet]
Output:
{{
    "A": {{
        "Sufficient": "Yes",
        "Answer": "De Smet"
    }},
    "R": "The author of The Long Winter is Laura Ingalls Wilder, and Laura Ingalls Wilder lived in De Smet."
}}

Q: Who is the coach of the team owned by Steve Bisciotti?
Memory: {{
    "1": "The triplets provide the information that Steve Bisciotti owns Baltimore Ravens.",
    "2": "No triplets provide the information."
}}
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, [Baltimore Ravens]
Steve Bisciotti, sports.sports_team_owner.teams_owned, [Baltimore Ravens]
Steve Bisciotti, organization.organization_founder.organizations_founded, [Allegis Group]
Output:
{{
    "A": {{
        "Sufficient": "No",
        "Answer": "Null"
    }},
    "R": "The team owned by Steve Bisciotti is Baltimore Ravens based on knowledge triplets. The coach of the team owned by Steve Bisciotti is not explicitly mentioned."
}}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Memory: {{
    "1": "The triplets provide the information that Rift Valley Province is located in Kenya.",
    "2": "The triplets provide the information that Kenya uses the Kenyan shilling as its currency."
}}
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
Output:
{{
    "A": {{
        "Sufficient": "Yes",
        "Answer": "Kenyan shilling"
    }},
    "R": "Based on knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency."
}}

Q: The country with the National Anthem of Bolivia borders which nations?
Memory: {{
    "1": "The triplets provide the information that the National Anthem of Bolivia is the anthem of Bolivia.",
    "2": "No triplets provide the information."
}}
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
Output:
{{
    "A": {{
        "Sufficient": "No",
        "Answer": "Null"
    }},
    "R": "Based on knowledge triplets, the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia. However, the given knowledge triplets do not provide information about which nations border Bolivia."
}}

Now you need to directly output the results of the following question in JSON format (must include "A" and "R") without other information or notes. If the triplets explicitly contains the answer to the question, prioritize the fact of the triplet over memory.
Q: {query}
Memory: {memory}
Knowledge Triplets: {relations}
Output Format (flat JSON): {{"A": {{"Sufficient": "Yes/No", "Answer": "A short string explaining the final answer."}}, "R": "reason"}}
Output:
"""

PROMPTS["answer"] = """
Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge. You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: I don't know. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: De Smet. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {{De Smet}}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: I don't know. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: Kenyan Shiling. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {{Kenyan shilling}}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

Q: {query}
Knowledge Triplets: {triplets}
A:
"""


PROMPTS["generate_directly"] = """You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers.
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {{Washington, D.C.}}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {{Bharoto Bhagyo Bidhata}}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {{Jason Allen Alexander}}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {{Peter Paul Rubens}}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {{Georgia}}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {{Heroin}}.

Q: {query}
A:
"""


# CONFIG PARAMETERS END---

@dataclass
class Query:
    query: str
    query_time: datetime = None
    subqueries: Optional[List[str]] = None
    memory: Optional[str] = None


class PoG_Model:
    """
    A reproduction of Plan-on-Graph model.
    """

    def __init__(
        self,
        domain: str = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "tog"
        self.domain = domain
        self.logger = logger

        self.width = 30
        self.depth = 3

    @llm_retry(max_retries=10, default_output=[])
    async def extract_entity(
        self,
        query: Query
    ) -> List[str]:

        system_prompt = PROMPTS["kg_topic_entity"]["system"]
        user_message = PROMPTS["kg_topic_entity"]["user"].format(
            query=query.query,
            query_time=query.query_time
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

        result = maybe_load_json(response)

        entities_list = []
        if result['domain'] == "movie":
            if result.get("movie_name"):
                if isinstance(result["movie_name"], str):
                    movie_names = result["movie_name"].split(",")
                else:
                    movie_names = result["movie_name"]
                for movie_name in movie_names:
                    entities_list.append(normalize_entity(movie_name))
            if result.get("person"):
                if isinstance(result["person"], str):
                    person_list = result["person"].split(",")
                else:
                    person_list = result["person"]
                for person in person_list:
                    entities_list.append(normalize_entity(person))
            if result.get("year"):
                if isinstance(result["year"], str) or isinstance(result["year"], int):
                    years = str(result["year"]).split(",")
                else:
                    years = result["year"]
                for year in years:
                    entities_list.append(normalize_entity(str(year)))
        elif result['domain'] == "sports":
            if result.get("tournament"):
                if isinstance(result["tournament"], str):
                    matches = result["tournament"].split(",")
                else:
                    matches = result["tournament"]
                for match in matches:
                    entities_list.append(normalize_entity(match))
            if result.get("team"):
                if isinstance(result["team"], str):
                    teams = result["team"].split(",")
                else:
                    teams = result["team"]
                for team in teams:
                    entities_list.append(normalize_entity(team))
        elif result['domain'] == "other":
            if result.get("main_entity"):
                if isinstance(result["main_entity"], str):
                    entities = result["main_entity"].split(",")
                else:
                    entities = result["main_entity"]
                for entity in entities:
                    entities_list.append(normalize_entity(entity))

        return entities_list

    @llm_retry(max_retries=10, default_output=[])
    async def align_topic(
        self,
        query: Query,
        topic_entities: List[str]
    ) -> List[RelevantEntity]:
        """
        Perform exact match in KG to align a list of topic entity strings of a query to KG entities.

        Args:
            query (str): The query itself.
            topic_entities (List[str]): A list of topic entity strings.
            top_k (int): Specify the top-k entities assessed in KG. Note that the search is based on approximate nearest-neighbor (ANN) search,
                        so, in general, a larger top_k retreive more accurate results.

        Returns:
            List[RelevantEntity]: A list of relevant KG entities with their relevant scores.
        """
        norm_coeff = 1 / len(topic_entities) if len(
            topic_entities) > 0 else 1  # Assuming all the topic entities are equally important
        results = []

        for idx, topic in enumerate(topic_entities):
            exact_match = kg_driver.get_entities(
                name=topic, top_k=1, fuzzy=True)
            results.append(RelevantEntity(exact_match[0], norm_coeff))

        return results

    @llm_retry(max_retries=10, default_output="[]")
    async def break_question(
        self,
        query: Query
    ) -> str:

        user_message = PROMPTS["subobjective_prompt"].format(
            query=query.query
        )
        response = await generate_response(
            [
                {"role": "user", "content": user_message},
            ],
            max_tokens=2048,
            logger=self.logger
        )
        self.logger.debug(user_message + "\n" + response)
        first_brace_p = response.find('[')
        last_brace_p = response.rfind(']')
        response = response[first_brace_p:last_brace_p+1]
        return response

    # Step 1

    @llm_retry(max_retries=10, default_output=[])
    async def relation_search_prune(
        self,
        query: Query,
        entity: KGEntity
    ) -> List[RelevantRelation]:

        width = self.width if self.width else PROMPTS["DEFAULT_WIDTH"]
        relation_list = kg_driver.get_relations(entity, unique_relation=True)
        if len(relation_list) == 0:
            return []
        entity_str = entity_to_text(entity)

        unique_relations_dict = {}
        for i, relation in enumerate(relation_list):
            relation.target = KGEntity(
                id="", type=relation.target.type, name="")
            relation.properties = {}
            unique_relations_dict[f"rel_{i}"] = relation

        unique_relations_str = "\n".join([
            f"{key}: {relation_to_text(relation,
                                       include_des=False,
                                       include_src_des=False,
                                       include_src_prop=False,
                                       property_key_only=True)}"
            for key, relation in unique_relations_dict.items()
        ])

        relevant_relations_score = {}
        system_prompt = PROMPTS["relations_pruning"]["system"].format(
            domain=self.domain,
            width=width
        )
        user_message = PROMPTS["relations_pruning"]["user"].format(
            query=query.query,
            query_time=query.query_time,
            subqueries=query.subqueries,
            entity=entity_str,
            relations=unique_relations_str
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2048,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + "\n" +
                          user_message + "\n" + response)

        relevant_relations_score = maybe_load_json(
            response)['relevant_relations']
        return [
            RelevantRelation(unique_relations_dict[ind], score)
            for ind, score in relevant_relations_score.items()
            if (score > 0) and (ind in unique_relations_dict)
        ]

    @llm_retry(max_retries=10, default_output=[])
    async def triplet_prune(
        self,
        query: Query,
        relevant_relation: RelevantRelation,
        triplet_candidates: List[KGRelation]
    ) -> List[RelevantEntity]:

        width = self.width if self.width else PROMPTS["DEFAULT_WIDTH"]
        triplet_dict = {
            f"rel_{i}": triplet for i, triplet in enumerate(triplet_candidates)
        }
        relations_str = "\n".join([
            f"{key}: {relation_to_text(triplet, include_src_prop=True, property_key_only=False)}"
            for key, triplet in triplet_dict.items()
        ])

        relevant_entities_score = {}
        system_prompt = PROMPTS["triplets_pruning"]["system"].format(
            domain=self.domain,
            width=width
        )
        user_message = PROMPTS["triplets_pruning"]["user"].format(
            query=query.query,
            query_time=query.query_time,
            relations=relations_str,
        )

        self.logger.debug("Triplet pruning...")
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2048,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(system_prompt + "\n" +
                          user_message + "\n" + response)

        relevant_entities_score = json.loads(response)['relevant_relations']
        return [
            RelevantRelation(triplet_dict[ind],
                             relevant_relation.score * score)
            for ind, score in relevant_entities_score.items()
            if (score > 0) and (ind in triplet_dict)
        ]

    def triplet_sort(
        self,
        total_relevant_triplets: List[RelevantRelation]
    ):
        width = self.width if self.width else PROMPTS["DEFAULT_WIDTH"]

        total_relevant_triplets = sorted(
            total_relevant_triplets, key=lambda x: x.score, reverse=True)[:width]
        filtered_relevant_triplets = [
            triplet for triplet in total_relevant_triplets if triplet.score > 0]

        cluster_chain_of_entities = [relation_to_text(
            triplet.relation) for triplet in filtered_relevant_triplets]

        return len(filtered_relevant_triplets) != 0, \
            cluster_chain_of_entities, \
            filtered_relevant_triplets

    def extract_memory(self, string):
        first_brace_p = string.find('{')
        last_brace_p = string.rfind('}')
        string = string[first_brace_p:last_brace_p+1]
        return string

    @llm_retry(max_retries=10, default_output=[])
    async def update_memory(
        self,
        query: Query,
        filtered_relevant_triplets: List[RelevantRelation]
    ) -> Query:
        relations_str = "\n".join([
            f"{relation_to_text(triplet.relation, include_src_prop=True, property_key_only=False)}"
            for triplet in filtered_relevant_triplets
        ])
        user_message = PROMPTS["update_mem_prompt"].format(
            query=query.query,
            query_time=query.query_time,
            subqueries=query.subqueries,
            memory=query.memory,
            relations=relations_str,
        )

        self.logger.debug("Memory updating...")
        response = await generate_response(
            [
                {"role": "user", "content": user_message}
            ],
            # max_tokens=2048,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(user_message + "\n" + response)

        query.memory = self.extract_memory(response)
        return query

    def extract_reason_and_anwer(self, string):
        first_brace_p = string.find('{')
        last_brace_p = string.rfind('}')
        string = string[first_brace_p:last_brace_p+1]
        answer = re.search(r'"Answer":\s*"(.*?)"', string)
        if answer:
            answer = answer.group(1)
        else:
            answer = re.search(r'"Answer":\s*(\[[^\]]+\])', string).group(1)

        reason = re.search(r'"R":\s*"(.*?)"', string).group(1)
        sufficient = re.search(r'"Sufficient":\s*"(.*?)"', string).group(1)
        return answer, reason, sufficient

    @llm_retry(max_retries=10, default_output=(False, ""))
    async def reasoning(
        self,
        query: Query,
        filtered_relevant_triplets: List[RelevantRelation]
    ):
        relations_str = "\n".join([
            f"{relation_to_text(triplet.relation, include_src_prop=True, property_key_only=False)}"
            for triplet in filtered_relevant_triplets
        ])
        user_message = PROMPTS["answer_depth_prompt"].format(
            query=query.query,
            query_time=query.query_time,
            memory=query.memory,
            relations=relations_str
        )

        response = await generate_response(
            [
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            logger=self.logger
        )
        self.logger.debug(user_message + '\n' + response)

        answer, reason, sufficient = self.extract_reason_and_anwer(response)
        return response, answer, sufficient

    @llm_retry(max_retries=10, default_output="I don't know.")
    async def answer(
        self,
        query: Query,
        topic_entities: List,
        cluster_chain_of_entities
    ) -> str:
        entities_str = '\n'.join(
            [f"ent_{idx}: {entity_to_text(entity)}" for idx, entity in enumerate(topic_entities)])
        entities_str = entities_str if entities_str else "None"
        idx = 0
        triplets = []
        for sublist in cluster_chain_of_entities:
            for chain in sublist:
                triplets.append(f"rel_{idx}: {chain}")
                idx += 1
        triplets_str = '\n'.join(triplets)
        triplets_str = triplets_str if triplets_str else "None"

        system_prompt = "You are an NLP expert in answering questions. Please help with the following task."
        user_message = PROMPTS["answer"].format(
            query=query.query,
            query_time=query.query_time,
            entities=entities_str,
            triplets=triplets_str,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            logger=self.logger
        )
        self.logger.debug(user_message + '\n' + response)

        return response

    async def half_stop(self, query: Query,
                        topic_entities,
                        cluster_chain_of_entities):
        print("No new knowledge added during search depth %d, stop searching." % self.depth)
        answer = await self.answer(query, topic_entities, cluster_chain_of_entities)
        return answer

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_without_explored_paths(
        self,
        query: Query
    ):
        system_prompt = "You are an NLP expert in answering questions. Please help with the following task."
        user_message = PROMPTS["generate_directly"].format(
            query=query.query,
            query_time=query.query_time,
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            logger=self.logger
        )
        self.logger.debug(system_prompt + '\n' +
                          user_message + '\n' + response)

        return response

    def extract_answer(self, text):
        start_index = text.find("{")
        end_index = text.find("}")
        if start_index != -1 and end_index != -1:
            return text[start_index+1:end_index].strip()
        else:
            return ""

    def if_true(self, prompt):
        if prompt.lower().strip().replace(" ", "") == "yes":
            return True
        return False

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        **kwargs
    ):
        query = Query(query=query, query_time=query_time)
        topic_entities = await self.extract_entity(query)
        self.logger.info(f"Extracted topic entities: {topic_entities}")
        topic_entities_scores = await self.align_topic(query, topic_entities)

        if len(topic_entities_scores) == 0:
            return await self.generate_without_explored_paths(query)

        query.sub_questions = await self.break_question(query)

        ans = ""
        cluster_chain_of_entities = []
        initial_topic_entities = [
            relevant_entity.entity for relevant_entity in topic_entities_scores]
        flag_printed = False

        for depth in range(1, self.depth + 1):
            tasks = [
                self.relation_search_prune(query, entity_score.entity)
                for entity_score in topic_entities_scores
                if entity_score.entity is not None
            ]
            results = await asyncio.gather(*tasks)  # Run tasks in parallel

            relevant_relations_list = []
            for entity_score, relevant_relations in zip(topic_entities_scores, results):
                relevant_relations_list.extend([
                    RelevantRelation(
                        relation=relevant_relation.relation,
                        score=relevant_relation.score * entity_score.score
                    )
                    for relevant_relation in relevant_relations
                ])

            tasks = []
            for relevant_relation in relevant_relations_list:
                triplet_candidates = kg_driver.get_relations(source=relevant_relation.relation.source,
                                                             relation=relevant_relation.relation.name,
                                                             target_type=relevant_relation.relation.target.type)

                if len(triplet_candidates) >= 120:
                    num_retain_entity = 120
                    triplet_candidates = random.sample(
                        triplet_candidates, num_retain_entity)

                if len(triplet_candidates) == 0:
                    continue

                # Store tasks and corresponding relation
                tasks.append(self.triplet_prune(query,
                                                relevant_relation,
                                                triplet_candidates))

            # Run all entity_score calls in parallel
            results = await asyncio.gather(*tasks)

            total_relevant_triplets = sum(results, [])

            flag, chain_of_entities, filtered_relevant_triplets = self.triplet_sort(
                total_relevant_triplets)
            cluster_chain_of_entities.append(chain_of_entities)

            norm_coeff = sum(
                triplet.score for triplet in filtered_relevant_triplets)
            norm_coeff = 1 / norm_coeff if norm_coeff > 0 else 1
            topic_entities_scores_dict = {}
            for triplet in filtered_relevant_triplets:
                last = topic_entities_scores_dict.setdefault(triplet.relation.target.id,
                                                             RelevantEntity(triplet.relation.target, 0))
                topic_entities_scores_dict[triplet.relation.target.id] = \
                    RelevantEntity(triplet.relation.target,
                                   triplet.score * norm_coeff + last.score)
            topic_entities_scores = list(topic_entities_scores_dict.values())

            if flag:
                query = await self.update_memory(query, filtered_relevant_triplets)

                results, answer, sufficient = await self.reasoning(query, filtered_relevant_triplets)
                if str(answer).lower() == 'null' or \
                   str(answer).lower() == 'none' or \
                   str(answer).startswith('m.') or \
                   str(answer).startswith('[\"m.') or \
                   str(answer).startswith("['m.") or \
                   'yes' not in str(sufficient).lower():
                    stop = False
                else:
                    stop = True

                if stop:
                    print("ToG stoped at depth %d." % depth)
                    ans = await self.half_stop(query, initial_topic_entities, cluster_chain_of_entities)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
            else:
                print(
                    "No new knowledge added during search depth %d, stop searching." % depth)
                ans = await self.half_stop(query, initial_topic_entities, cluster_chain_of_entities)
                flag_printed = True
                break

        if not flag_printed:
            ans = await self.generate_without_explored_paths(query)

        return ans
