import asyncio
import json
import textwrap
from typing import Any, Dict, List

from inference import *
from kg.kg_driver import *
from kg.kg_rep import *
from utils.prompt_list import *
from utils.utils import *
from utils.logger import *

PROMPTS = get_default_prompts()

PROMPTS["break_down_question"] = {
    "system": textwrap.dedent("""\
    You are a helpful assistant who is good at answering questions in the {domain} domain by using knowledge from an external knowledge graph. Before answering the question, you need to break down the question
    so that you may look for the information from the knowledge graph in a step-wise operation. Hence, please break down the process of answering the question into as few sub-objectives as possible based on semantic analysis. 
    A query time is also provided; please consider including the time information when applicable.
    
    There can be multiple possible route to break down the question, aim for generating {route} possible routes. Note that every route may have a different solving efficiency, order the route by their solving efficiency.
    Return your reasoning and sub-objectives as multiple lists of strings in a flat JSON of format: {{"reason": "...", "routes": [[<a list of sub-objectives>], [<a list of sub-objectives>], ...]}}. (TIP: You will need to escape any double quotes in the string to make the JSON valid)

    Domain-specific Hints:
    {hints}

    -Example-
    Q: Which of the countries in the Caribbean has the smallest country calling code?
    Query Time: 03/05/2024, 23:35:21 PT
    Output: {{
    "reason": "The most efficient route involves directly identifying Caribbean countries and their respective calling codes, as this limits the scope of the search. In contrast, routes that involve broader searches, such as listing all country calling codes worldwide before filtering, are less efficient due to the larger dataset that needs to be processed. Therefore, routes are ordered based on the specificity of the initial search and the subsequent steps required to narrow down to the answer.",
    "routes": [["List all Caribbean countries", "Determine the country calling code for each country", "Identify the country with the smallest calling code"],
               ["Identify Caribbean countries", "Retrieve their country calling codes", "Compare to find the smallest"],
               ["Identify the smallest country calling code globally", "Filter by Caribbean countries", "Select the smallest among them"],
               ["List all country calling codes worldwide", "Filter the calling codes by Caribbean countries", "Find the smallest one"]]
    }}
    """),

    "user": textwrap.dedent("""\
    Q: {query}
    Query Time: {query_time}
    Output:""")
}

PROMPTS["relations_pruning"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are given a question in the {domain} domain, its query time, a potential route to solve it, an entity, and a list of relations starting from it.
    The goal is to retrieve up to {width} relations that contribute to answering the steps in the solving route and, therefore, answer the question. Rate their relevance from 0 to 1 (the sum of the scores of these relations is 1).
    
    -Steps-
    1. You are provided a list of directed relations between entities in the format of
    rel_i: (entity_type: entity_name)-[relation_type, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(entity_type: entity_name).
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
    
    2. Retrieve relations only from the given list that contribute to answering the question, and provide a short reason for your scoring.
    Return its index (rel_i) and score into a json of the format: {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}.
    (TIP: You will need to escape any double quotes in the string to make the JSON valid)
    
    Domain-specific Hints:
    {hints}
    
    ######################
    -Examples-
    Question: Which movie wins the best visual effect award in 2006 Oscars?
    Solving Route: ["Identify the 2006 Oscars best visual effects winner directly from the knowledge graph"]
                              
    Entity: (Award: VISUAL EFFECTS, properties: <year: 2006, ceremony_number: 78, type: OSCAR AWARD>)
    Relations: rel_0: (Award: VISUAL EFFECTS)-[HELD_IN]->(Year: None)
    rel_1: (Award: VISUAL EFFECTS)-[NOMINATED_FOR, properties: <winner, person, movie>]->(Movie: None)
    rel_2: (Award: VISUAL EFFECTS)-[WON, properties: <winner, person, movie>]->(Movie: None)
    Output: {{"reason": "The question is asking for movies that won the award, relation rel_2 is the most relevant to award winning. rel_1 is relation that find movies released in 2006\
    and may help find the movie that wins the award. A movie that won the award should also got nominated for the award, so rel_1 also has slight relevance. ", 
    "relevant_relations": {{"rel_2": 0.7, "rel_0": 0.2, "rel_1": 0.1}}
    }}
    #####
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    
    Entity: {entity}
    Relations: {relations}
    
    Output Format (flat JSON): {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}.
    Output:
    """)
}

PROMPTS["triplets_pruning"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are presented with a question in the {domain} domain, its query time, a potential route to solve it.
    You will then given a source entity (type, name, description, and potential properties) and a list of directed relations starting from / ended at the source entity in the format of (source entity)-[relation]->(target entity).
    The goal is to score the relations' contribution to answering the steps in the solving route and, therefore, answer the question. Rate them on a scale from 0 to 1 (the sum of the scores of all relations is 1).
    
    -Steps-
    1. You are provided the source entity in the format of "(source_entity_type: source_entity_name, desc: "description", props: {{key1: val, key2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})"
    where the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
                              
    2. You are then provided a list of directed relations in the format of
    "rel_i: (source_entity_type: source_entity_name)-[relation_type, desc: "description", props: {{key1: val, key2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(entity_type: entity_name, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})"
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.
    You are going to assess the relevance of the relation type and its properties, along with the target entity name and its properties, to the given question.
    
    3. Score the relations' relevance to answering the question, and provide a short reason for your scoring.
    Return its index (ent_i) and score into a valid JSON of the format: {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}.
    (TIP: You will need to escape any double quotes in the string to make the JSON valid)
                              
    Domain-specific Hints:
    {hints}
    
    ##### Examples #####
    Question: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
    Query Time: 03/19/2024, 22:59:20 PT
    Solving Route: ["List movies produced by Tobin Armbrust", "Filter by movies featuring Miley Cyrus", "Identify the movie"]
                              
    Source Entity: (Person: Tobin Armbrust)
    Relations: rel_0: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: The Resident)
    rel_1: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: So Undercover, properties: <featured: Miley Cyrus, Jeremy Piven, and Mike O'Malley>)
    rel_2: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: Let Me In, properties: <featured: Kodi Smit-McPhee, Chloë Grace Moretz, Elias Koteas, Cara Buono, and Richard Jenkins>)
    rel_3: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: Begin Again, properties: <featured: Keira Knightley, Mark Ruffalo, Hailee Steinfeld>)
    rel_4: (Person: Tobin Armbrust)-[PRODUCED]->(Movie: A Walk Among the Tombstones, properties: <featured: Liam Neeson, Dan Stevens, David Harbour>)
    Output: {{"reason": "The movie that matches the given criteria is 'So Undercover' with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for 'So Undercover' would be 1, and the scores for all other entities would be 0.", "relevant_relations": {{"rel_1": 1.0}}}}
    ####
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
                            
    Source Entity: {entity}
    Relations: {relations}
    
    Output Format (flat JSON): {{"reason": "reason", "relevant_relations": {{"rel_i": score_i, "rel_i": score_j, ...}}}}
    Output:
    """)
}

PROMPTS["evaluate"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are presented with a question in the {domain} domain, its query time, and a potential route to solve it. \
    Given the retrieved related entities and triplets from a noisy knowledge graph, you are asked to determine whether these references and your knowledge are sufficient to answer the question (Yes or No).
    - If yes, answer the question using fewer than 50 words.
    - If no, respond with 'I don't know'.
                              
    1. The entities will be given in the format of
    "ent_i: (<entity_type>: <entity_name>, desc: "description", props: {{key_1: val, key_2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})"
    The triplets will be given in the format of
    "rel_i: (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {{key_1: val, key_2: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(<target_entity_type>: <target_entity_name>)"
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, "props" are associated properties of the entity or relation. 
    Each property may have a single value, or multiple valid values of vary confidence under different context. The percentage is confidence score, and "ctx" is the optional context under which the value is valid.
    If multiple conflicting candidates are found, use the one with stronger supporting evidence such as temporal-aligned triplets or consists of additional supporting properties. If a more strongly justified answer exists, prefer it.

    2. Return your judgment in a JSON of the format {{"sufficient": "Yes/No", "reason": "...", "answer": "..."}} (TIP: You will need to escape any double quotes in the string to make the JSON valid)
    
    Domain-specific Hints:
    {hints}
                              
    #### Examples ####
    Question: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
    Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
    Output: {{"sufficient": "No",
              "reason": "Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said 'Taste cannot be controlled by law,' which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.",
              "answer": "I don't know."}}

    Question: The artist nominated for The Long Winter lived where?
    Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
    Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
    Unknown-Entity, people.place_lived.location, De Smet
    Output: {{"sufficient": "Yes",
              "reason": "Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is De Smet.",
              "answer": "De Smet."}}

    Question: Who is the coach of the team owned by Steve Bisciotti?
    Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
    Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
    Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
    Output: {{"sufficient": "No",
              "reason": "Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.",
              "answer": "I don't know."}}

    Question: Rift Valley Province is located in a nation that uses which form of currency?
    Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
    Rift Valley Province, location.location.geolocation, UnName_Entity
    Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
    Kenya, location.country.currency_used, Kenyan shilling
    Output: {{"sufficient": "Yes",
              "reason": "Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is Kenyan shilling.",
              "answer": "Kenyan shilling."}}

    Question: The country with the National Anthem of Bolivia borders which nations?
    Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
    National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
    National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
    UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
    Bolivia, location.country.national_anthem, UnName_Entity
    Output: {{"sufficient": "No",
              "reason": "Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.",
              "answer": "I don't know."}}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Solving Route: {route}
    Knowledge Entities: {entities}
    Knowledge Triplets: {triplets}
    
    Output Format (flat JSON): {{"sufficient": "Yes/No", "reason": "...", "answer": "..."}}
    Output:
    """)
}

PROMPTS["validation"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are presented with a question in the {domain} domain, and its query time. The goal is to answer the question *accurately* - you will be rewarded for correctly answering the question, *penalized* by providing a wrong answer. 

    A confident but careless friend has provided us a tentative answer, denote as "attempt". We don't really trust it, so we have identified a list of potential routes to solve it. So far, we have followed a portion of the routes, retrieved a list of potential associated retrieved knowledge graph entities and triplets (entity, relation, entity), and provided tentative answers.
    The entities will be given in the format of
    "ent_i: (<entity_type>: <entity_name>, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}})"
    The triplets will be given in the format of
    "rel_i: (<source_entity_type>: <source_entity_name>)-[<relation_name>, desc: "description", props: {{key: [val_1 (70%, ctx:"context"), val_2 (30%, ctx:"context")], ...}}]->(<target_entity_type>: <target_entity_name>)"
    where "i" is the index, arrow symbol ("->" or "<-") is the relation direction, the percentage is confidence score, "ctx" is an optional context under which the value is valid. Each property may have only a single value, or multiple valid values of vary confidence under different context.

    You will act as a rigorous judge to whether the answers reach a consensus or not before running out of solving routes. Consensus is defined by at least a half of the answers (including my friend's attempt) agree on a specific answer.
    Please exactly follow these strategies to guarantee that your answer will perform at least better than my friend:

    1. If there is a consensus, then respond with "Yes", and summarize them into a final answer following with a summarized explanation. 
    
    2. If there is not consensus, and there are still unexplored solving routes, then respond with "No", and don't provide a final answer. We will continue exploring the next solving route.
    
    3. If there is not consensus, and we run out of unexplored solving route, you have to respond with "Yes", and summarize them into a final answer following with a summarized explanation.
    If multiple conflicting answers are found, use the one with more votes (consensus), stronger supporting evidence such as temporal-aligned triplets or consists of additional supporting properties. If a more strongly justified answer exists, prefer it.
    
    4. Lastly, if none of the solving routes give a resonable answer (all "I don't know"), then fall back to use my friend's attempt.                           
    
    If the references do not contain the necessary information to answer the question, respond with 'I don't know'.
    Remember, you will be rewarded for correctly answering the question, penalized by providing a wrong answer. There is no reward or penalty if you answer "I don't know", which is more preferable than providing a wrong answer.                   
    
    Please return the output in a JSON of the format: {{"judgement": "Yes/No", "final_answer": "\"The Final Answer\". A short explanation of how to interpret the final answer."}}
    
    Domain-specific Hints:
    {hints}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    Attempt: {attempt}
    """)
}

PROMPTS["generate_directly"] = {
    "system": textwrap.dedent("""\
    -Goal-
    You are provided with a question in the {domain} domain, and its query time. You are asked to determine whether your knowledge are sufficient to answer the question (Yes or No).
    - If yes, answer the question succinctly, using the fewest words possible. 
    - If no, respond with 'I don't know'.
    Please explain your reasoning and provide supporting evidence from your knowledge to support your answer.
                              
    Return your judgment in a JSON of the format {{"sufficient": "Yes/No", "reason": "...", "answer": "..."}} (TIP: You will need to escape any double quotes in the string to make the JSON valid)

    #### Examples ####
    Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
    Output: {{"sufficient": "Yes",
              "reason": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is Washington, D.C.",
              "answer": "Washington, D.C."}}

    Question: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
    Output: {{"sufficient": "Yes",
              "reason": "First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is Bharoto Bhagyo Bidhata.",
              "answer": "Bharoto Bhagyo Bidhata"}}
                              

    Question: Who was the artist nominated for an award for You Drive Me Crazy?
    Output: {{"sufficient": "Yes",
              "reason": "First, the song 'You Drive Me Crazy' was performed by Britney Spears. Second, Britney Spears was nominated for awards for this song. The answer is Britney Spears.",
              "answer": "Britney Spears"}}
                              

    Question: What person born in Siegen influenced the work of Vincent Van Gogh?
    Output: {{"sufficient": "Yes",
              "reason": " First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is Peter Paul Rubens.",
              "answer": "Peter Paul Rubens"}}
                             

    Question: What is the country close to Russia where Mikheil Saakashvii holds a government position?
    Output: {{"sufficient": "Yes",
              "reason": "First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is Georgia.",
              "answer": "Georgia"}}
                              

    Question: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
    Output: {{"sufficient": "Yes",
              "reason": "First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is Heroin.",
              "answer": "Heroin"}}
    """),

    "user": textwrap.dedent("""\
    Question: {query}
    Query Time: {query_time}
    
    Output Format (flat JSON): {{"sufficient": "Yes/No", "reason": "...", "answer": "..."}}
    Output:
    """)
}


@dataclass
class Query:
    query: str
    query_time: datetime = None
    subqueries: Optional[List[str]] = None


class OurMinusGlobalSearch_Model:
    """
    Our proposed model.
    """

    def __init__(
        self,
        domain: str = None,
        config: dict = None,
        logger: BaseProgressLogger = DefaultProgressLogger(),
        **kwargs
    ):
        self.name = "our_minus_global_search"
        self.domain = domain
        self.logger = logger

        self.config = {
            "route": 5,
            "width": 30,
            "depth": 3
        }
        if config:
            self.config.update(config)

        self.route = self.config["route"]
        self.width = self.config["width"]
        self.depth = self.config["depth"]
        print(self.config)

    @llm_retry(max_retries=10, default_output=[])
    async def break_down_question(
        self,
        query: Query
    ) -> List[Query]:
        system_prompt = PROMPTS["break_down_question"]["system"].format(
            domain=self.domain,
            route=self.route,
            hints=PROMPTS["domain_hints"][self.domain]
        )
        user_message = PROMPTS["break_down_question"]["user"].format(
            query=query.query,
            query_time=query.query_time
        )

        response = await generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=2048,
            # DeepSeek-V3 generates endless JSON with json_object enforcement, has to turn it off
            response_format={
                "type": "json_object"} if "deepseek" not in MODEL_NAME.lower() else None,
            logger=self.logger
        )
        self.logger.debug(user_message + '\n' + response)

        routes = maybe_load_json(response)["routes"]
        queries = []
        for route in routes:
            queries.append(Query(
                query=query.query,
                query_time=query.query_time,
                subqueries=route
            ))
        return queries

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
            hints=PROMPTS["domain_hints"][self.domain],
            width=width
        )
        user_message = PROMPTS["relations_pruning"]["user"].format(
            query=query.query,
            query_time=query.query_time,
            route=query.subqueries,
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
        self.logger.debug(user_message + "\n" + response)

        relevant_relations_score = maybe_load_json(
            response)['relevant_relations']
        return [
            RelevantRelation(unique_relations_dict[ind], float(score))
            for ind, score in relevant_relations_score.items()
            if (float(score) > 0) and (ind in unique_relations_dict)
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
            f"rel_{i}": triplet for i, triplet in enumerate(triplet_candidates[:min(width, len(triplet_candidates))])
        }
        source = list(triplet_dict.values())[0].source
        entity_str = entity_to_text(source)
        relations_str = "\n".join([
            f"{key}: {relation_to_text(triplet, include_src_des=False, include_src_prop=False)}"
            for key, triplet in triplet_dict.items()
        ])
        if len(triplet_dict) < len(triplet_candidates):
            relations_str += f"\n...({len(triplet_candidates) - len(triplet_dict)} relation(s) truncated)"

        relevant_entities_score = {}
        system_prompt = PROMPTS["triplets_pruning"]["system"].format(
            domain=self.domain,
            hints=PROMPTS["domain_hints"][self.domain],
            width=width
        )
        user_message = PROMPTS["triplets_pruning"]["user"].format(
            query=query.query,
            query_time=query.query_time,
            route=query.subqueries,
            entity=entity_str,
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
        self.logger.debug(user_message + "\n" + response)

        relevant_entities_score = json.loads(response)['relevant_relations']
        return [
            RelevantRelation(triplet_dict[ind],
                             relevant_relation.score * float(score))
            for ind, score in relevant_entities_score.items()
            if (float(score) > 0) and (ind in triplet_dict)
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
        route: Query,
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

        system_prompt = PROMPTS["evaluate"]["system"].format(
            domain=self.domain,
            hints=PROMPTS["domain_hints"][self.domain]
        )
        user_message = PROMPTS["evaluate"]["user"].format(
            query=route.query,
            query_time=route.query_time,
            route=route.subqueries,
            entities=entities_str,
            triplets=triplets_str
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
        self.logger.debug(user_message + '\n' + response)

        result = maybe_load_json(response)
        return result["sufficient"].lower().strip().replace(" ", "") == "yes", \
            result.get("reason", ""), \
            result.get("answer", "I don't know.")

    @llm_retry(max_retries=10, default_output="I don't know.")
    async def answer(
        self,
        route: Query,
        topic_entities: List,
        cluster_chain_of_entities: List,
        reason: str
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

        system_prompt = PROMPTS["answer"]["system"].format(
            domain=self.domain
        )
        user_message = PROMPTS["answer"]["user"].format(
            query=route.query,
            query_time=route.query_time,
            route=route.subqueries,
            entities=entities_str,
            triplets=triplets_str,
            reason=reason
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

    @llm_retry(max_retries=10, default_output=(False, ""))
    async def validation(
        self,
        queries: List[Query],
        attempt: str,
        results: List
    ):
        system_prompt = PROMPTS["validation"]["system"].format(
            domain=self.domain,
            hints=PROMPTS["domain_hints"][self.domain]
        )
        user_message = PROMPTS["validation"]["user"].format(
            query=queries[0].query,
            query_time=queries[0].query_time,
            attempt=attempt
        )
        user_message += f"\nWe have identified {len(queries)} solving route(s) below, and have {len(queries) - len(results)} unexplored solving route left.:\n"
        for idx in range(len(results)):
            user_message += f"Route {idx + 1}: {queries[idx].subqueries}\n" + \
                "Reference: " + results[idx]["context"] + '\n' + \
                "Answer: " + results[idx]["ans"] + '\n\n'
        for idx in range(len(results), len(queries)):
            user_message += f"Route {idx + 1}: {queries[idx].subqueries}\n\n"
        user_message += 'Output Format (flat JSON): {"judgement": "Yes/No", "final_answer": "<Your Final Answer>. <A short explanation of how to interpret the final answer>"}'
        user_message += "Output:"

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

        result = maybe_load_json(response)

        return result["judgement"].lower().strip().replace(" ", "") == "yes", result.get("final_answer", "")

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_without_explored_paths(
        self,
        query: Query
    ):
        system_prompt = PROMPTS["generate_directly"]["system"].format(
            domain=self.domain
        )
        user_message = PROMPTS["generate_directly"]["user"].format(
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
        self.logger.debug(user_message + '\n' + response)

        result = maybe_load_json(response)
        return result.get("reason", ""), \
            result.get("answer", "I don't know.")

    @llm_retry(max_retries=10, default_output=("I don't know."))
    async def generate_answer(
        self,
        query: str,
        query_time: datetime = None,
        return_details: bool = False,
        **kwargs
    ) -> str:
        query = Query(query=query, query_time=query_time)
        query_embedding = (await generate_embedding([query.query], logger=self.logger))[0]
        reason, attempt = await self.generate_without_explored_paths(query)
        attempt = f'"{attempt}". {reason}'

        queries = await self.break_down_question(query)

        stop = False
        final = ""
        route_results = []
        for route in queries:
            topic_entities = await self.extract_entity(route)
            self.logger.info(f"Extracted topic entities: {topic_entities}")

            topic_entities_scores = await self.align_topic(route, topic_entities)

            ans = ""
            cluster_chain_of_entities = []
            initial_topic_entities = [
                relevant_entity.entity for relevant_entity in topic_entities_scores]

            all_entities = {}
            all_relations = {}
            for relevant_entity in topic_entities_scores:
                relevant_entity.step = 0
                all_entities[relevant_entity.entity.id] = relevant_entity

            stop, reason, answer = await self.reasoning(route, initial_topic_entities, [[]])
            if stop:
                print("ToG stoped at depth 0.")
                # await self.answer(route, initial_topic_entities, [[]], reason)
                ans = answer
            else:
                for depth in range(1, self.depth + 1):
                    tasks = [
                        self.relation_search_prune(route,
                                                   entity_score.entity)
                        for entity_score in topic_entities_scores
                        if entity_score.entity is not None
                    ]
                    # Run tasks in parallel
                    results = await asyncio.gather(*tasks)

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
                        # Query embedding based reranking
                        triplet_candidates = kg_driver.get_relations(source=relevant_relation.relation.source,
                                                                     relation=relevant_relation.relation.name,
                                                                     target_type=relevant_relation.relation.target.type,
                                                                     target_embedding=query_embedding)

                        # Filter visited triplets
                        triplet_candidates = [
                            triplet
                            for triplet in triplet_candidates
                            if triplet.id not in all_relations
                        ]

                        if len(triplet_candidates) == 0:
                            continue

                        # Store tasks and corresponding relation
                        tasks.append(self.triplet_prune(route,
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
                        stop, reason, answer = await self.reasoning(route, initial_topic_entities, cluster_chain_of_entities)
                        if stop:
                            print("ToG stoped at depth %d." % depth)
                            # await self.answer(route, initial_topic_entities, cluster_chain_of_entities, reason)
                            ans = answer
                            break
                        else:
                            print("depth %d still not find the answer." % depth)
                            ans = reason
                    else:
                        print(
                            "No new knowledge added during search depth %d, stop searching." % depth)
                        # ans = await self.answer(route, initial_topic_entities, cluster_chain_of_entities, "")
                        _, _, ans = await self.reasoning(route, initial_topic_entities, cluster_chain_of_entities)
                        break

            entities_str = '\n'.join(
                [f"ent_{idx}: {entity_to_text(entity)}" for idx, entity in enumerate(initial_topic_entities)])
            entities_str = entities_str if entities_str else "None"
            idx = 0
            triplets = []
            for sublist in cluster_chain_of_entities:
                for chain in sublist:
                    triplets.append(f"rel_{idx}: {chain}")
                    idx += 1
            triplets_str = '\n'.join(triplets)
            triplets_str = triplets_str if triplets_str else "None"
            route_results.append({
                "query": route,
                "context": "Knowledge Entities:\n" + entities_str + '\n' +
                           "Knowledge Triplets:\n" + triplets_str,
                "ans": f'"{ans}". {reason}',
                "entities": list(all_entities.values()),
                "relations": list(all_relations.values())
            })

            if len(route_results) >= 2:
                stop, final = await self.validation(queries, attempt, route_results)
                if stop:
                    print(final)
                    break

        if not stop:
            final = attempt

        if return_details:
            return final, route_results
        else:
            return final
