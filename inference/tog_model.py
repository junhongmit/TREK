import asyncio
import json
import random
import textwrap
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


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# VLLM Parameters
# TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_TENSOR_PARALLEL_SIZE = 8
# TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.9

# Sentence Transformer Parameters
# TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128

PROMPTS = get_default_prompts()

# From CRAG Benchmark: https://github.com/facebookresearch/CRAG/blob/main/models/rag_knowledge_graph_baseline.py
PROMPTS['kg_topic_entity'] = {
    "system": textwrap.dedent("""\
    You are given a Query and Query Time. Do the following: 

    1) Determine the domain the query is about. The domain should be one of the following: "sports", "movie", and "other". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

    2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

    For `movie` queries, these are possible keys:
    - `movie_name`: name of the movie
    - `person`: person name related to moves
    - `year`: if the query is about movies released in a specific year, extract the year

    For `sports` queries, these are possible keys:
    - `sport_type`: one of `basketball`, `soccer`, `other`
    - `tournament`: such as NBA, World Cup, Olympic.
    - `team`: teams that user interested in.
    - `datetime`: time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. 

    For `other` queries, these are possible keys:
    -  `main_entity`: extract the main entity of the query. 

    Return the results in a FLAT json. 

    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*
    
    EXAMPLE JSON OUTPUT:
    {"domain": "movie", "movie_name": "Mount Everest"}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    EXAMPLE JSON OUTPUT:
    {{"domain": "movie", "movie_name": "Mount Everest"}}
    Output:
    """)
}

PROMPTS["align_topic"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are presented with a question in the {domain} domain, its query time, and a list of entities extracted from a noisy knowledge graph.
    The goal is to identify all possible relevant entities to answering the question.
    The entities' relevance would be scored on a scale from 0 to 1 (use at most 3 decimal places, and remove trailing zeros; the sum of the scores of all entities is 1). 
    
    -Steps-
    1. You are provided a set of entities (type, name, description, and potential properties) globally searched from a knowledge graph that most similar to the question description, but may not be directly relevant to the question itself.
    Given in the format of "ent_i: (<entity_type>: <entity_name>, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})"
    where "i" is the index, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
    
    2. Score *ALL POSSIBLE* entities that are relevant to answering the question, and provide a short reason for your scoring.
    Return its index (ent_i) and score into a valid JSON of the format: {{"reason": "reason", "relevant_entites": {{"ent_i": 0.6, "ent_j": 0.3, ...}}}}. (TIP: You will need to escape any double quotes in the string to make the JSON valid)
    
    *NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  

    ######################
    -Examples-
    Question: How many countries were "Inception" filmed in?
    Query Time: 03/05/2024, 23:35:21 PT
    Entities: ent_0: (Movie: INCEPTION, desc: 2010 sci-fi action film, props: {{year: 2010, release_date: 2012-07-20, rating: 8.6}})
    ent_1: (Movie: INCEPTION: THE COBOL JOB, props: {{release_date: 2010-12-07, rating: 7.263, original_name: Inception: The Cobol Job}})
    ent_2: (Movie: INVASION, props: {{release_date: 2005-10-06, original_name: Invasion}})
    ent_3: (Movie: THE INVITATION, props: {{release_date: 2016-04-08, rating: 6.462, original_name: The Invitation}})
    Output: {{"reason": "The question is asking about the filming locations of the movie 'Inception', and ent_0 is the entity that directly corresponds to the movie 'Inception'.", "relevant_entites": {{"ent_0": 1}}}}

    Question: In this year, which animated film was recognized with the best animated feature film Oscar?
    Query Time: 03/19/2024, 23:49:30 PT
    Entities: ent_0: (Award: ANIMATED FEATURE FILM, props: {{year: 2024, ceremony_number: 96, type: OSCAR AWARD}})
    ent_1: (Award: SHORT FILM (ANIMATED), props: {{year: 2004, ceremony_number: 76, type: OSCAR AWARD}})
    ent_2: (Award: ANIMATED FEATURE FILM, props: {{year: 2005, ceremony_number: 77, type: OSCAR AWARD}})
    ent_3: (Award: ANIMATED FEATURE FILM, props: {{year: 2002, ceremony_number: 74, type: OSCAR AWARD}})
    ent_4: (Award: ANIMATED FEATURE FILM, props: {{year: 2003, ceremony_number: 75, type: OSCAR AWARD}})
    Output: {{"reason": "The entity ent_0 is the award for the best animated feature film in the year of query time, 2024.", "relevant_entities": {{"ent_0": 1}}}}

    Question: Can you tell me the name of the actress who starred in the film that won the best picture oscar in 2018?
    Query Time: 03/19/2024, 22:59:20 PT
    Entities: ent_0: (Award: ACTRESS IN A LEADING ROLE, props:{{year: 2018, ceremony_number: 90, type: OSCAR AWARD}})
    ent_1: (Award: ACTOR IN A LEADING ROLE, props: {{year: 2018, ceremony_number: 90, type: OSCAR AWARD}})
    ent_2: (Award: BEST PICTURE, props: {{year: 2018, ceremony_number: 90, type: OSCAR AWARD}})
    ent_3: (Award: ACTRESS IN A SUPPORTING ROLE, props: {{year: 2018, ceremony_number: 90, type: OSCAR AWARD}})
    Output:{{"reason": "The question is asking about the 2018 best picture Oscar movies, and award ent_2 is for the best picture in 2018. The award ent_0 is for the actress in a leading role in 2018, which may also help answer the question.", "relevant_entities": {{"ent_2": 0.8, "ent_0": 0.2}}}}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Entities: {top_k_entities_str}
    Output:
    """)
}

PROMPTS["relations_pruning"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are given a question in the {domain} domain, its query time, an entity, and a list of relations starting from it.
    The goal is to retrieve up to {width} relations that contribute to answering the question and rate their relevance from 0 to 1 (use at most 3 decimal places, and remove trailing zeros; the sum of the scores of these relations is 1).

    -Steps-
    1. You are provided a list of directed relations between entities in the format of
    rel_i: (entity_type: entity_name)-[relation_type, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(entity_type: entity_name).
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.

    2. Retrieve relations only from the given list that contribute to answering the question, and provide a short reason for your scoring.
    Return its index (rel_i) and score into a json of the format: {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}.
    (TIP: You will need to escape any double quotes in the string to make the JSON valid)
    
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
    Entity: {entity}
    Relations: {relations}
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
    Output:
    """)
}

PROMPTS["evaluate"] = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {{No}}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {{Yes}}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {{De Smet}}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {{No}}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {{Yes}}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {{Kenyan shilling}}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {{No}}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

Q: {query}
Knowledge Triplets: {triplets}
A:
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


class ToG_Model:
    """
    A reproduction of Think-on-Graph model.
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
        query: str,
        query_time: datetime
    ) -> List[str]:
        system_prompt = PROMPTS["kg_topic_entity"]["system"]
        user_message = PROMPTS["kg_topic_entity"]["user"].format(
            query=query,
            query_time=query_time
        )

        # Run all requests asynchronously
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
        query: str,
        query_time: datetime,
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

    # Step 1
    @llm_retry(max_retries=10, default_output=[])
    async def relation_search_prune(
        self,
        query: str,
        query_time: datetime,
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
            query=query,
            query_time=query_time,
            entity=entity_str,
            relations=unique_relations_str
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=4096,
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
        query: str,
        query_time: datetime,
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
            query=query,
            query_time=query_time,
            relations=relations_str,
        )

        self.logger.debug("Triplet pruning...")
        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=4096,
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

    @llm_retry(max_retries=10, default_output=(False, ""))
    async def reasoning(
        self,
        query: str,
        query_time: datetime,
        topic_entities,
        cluster_chain_of_entities
    ):
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

        system_prompt = "You are an NLP expert in analysing relations in text. Please help with the following task."
        user_message = PROMPTS["evaluate"].format(
            query=query,
            query_time=query_time,
            entities=entities_str,
            triplets=triplets_str
        )

        self.logger.debug(user_message)

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            logger=self.logger
        )
        self.logger.debug(response)

        result = self.extract_answer(response)
        # print(result)
        if self.if_true(result):
            return True, response
        else:
            return False, response

    @llm_retry(max_retries=10, default_output="I don't know.")
    async def answer(
        self,
        query: str,
        query_time: datetime,
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
            query=query,
            query_time=query_time,
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

    async def half_stop(
        self,
        query: str,
        query_time: datetime,
        topic_entities,
        cluster_chain_of_entities
    ):
        print("No new knowledge added during search depth %d, stop searching." % self.depth)
        answer = await self.answer(query, query_time, topic_entities, cluster_chain_of_entities)
        return answer

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_without_explored_paths(
        self,
        query: str,
        query_time: datetime
    ):
        system_prompt = "You are an NLP expert in answering questions. Please help with the following task."
        user_message = PROMPTS["generate_directly"].format(
            query=query,
            query_time=query_time,
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
        topic_entities = await self.extract_entity(query, query_time)
        self.logger.info(f"Extracted topic entities: {topic_entities}")

        topic_entities_scores = await self.align_topic(query, query_time, topic_entities)
        ans = ""
        cluster_chain_of_entities = []
        initial_topic_entities = [
            relevant_entity.entity for relevant_entity in topic_entities_scores]
        flag_printed = False
        all_entities = {}
        all_relations = {}
        for relevant_entity in topic_entities_scores:
            relevant_entity.step = 0
            all_entities[relevant_entity.entity.id] = relevant_entity
        stop, results = await self.reasoning(query, query_time, initial_topic_entities, [[]])
        if stop:
            print("ToG stoped at depth 0.")
            ans = await self.half_stop(query, query_time, initial_topic_entities, cluster_chain_of_entities)
        else:
            for depth in range(1, self.depth + 1):
                tasks = [
                    self.relation_search_prune(query, query_time,
                                               entity_score.entity)
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

                    # entity_candidates = await self.entity_score(entity_candidates, query, relation['score'], relation['relation'])
                    # print(entity_candidates)
                    # total_candidates.extend(update_history(relation, entity_candidates))

                    # Store tasks and corresponding relation
                    tasks.append(self.triplet_prune(query, query_time,
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
                topic_entities_scores = list(
                    topic_entities_scores_dict.values())

                for relevant_relation in filtered_relevant_triplets:
                    relevant_relation.relation.step = depth
                    all_relations[relevant_relation.relation.id] = relevant_relation
                for relevant_entity in topic_entities_scores:
                    relevant_entity.step = depth
                    all_entities[relevant_entity.entity.id] = relevant_entity
                if flag:
                    stop, results = await self.reasoning(query, query_time, initial_topic_entities, cluster_chain_of_entities)
                    if stop:
                        print("ToG stoped at depth %d." % depth)
                        ans = await self.half_stop(query, query_time, initial_topic_entities, cluster_chain_of_entities)
                        flag_printed = True
                        break
                    else:
                        print("depth %d still not find the answer." % depth)
                else:
                    print(
                        "No new knowledge added during search depth %d, stop searching." % depth)
                    ans = await self.half_stop(query, query_time, initial_topic_entities, cluster_chain_of_entities)
                    flag_printed = True
                    break

            if not flag_printed:
                ans = await self.generate_without_explored_paths(query, query_time)

        return ans
