from inference.io_model import IO_Model
from inference.cot_model import CoT_Model
from inference.sc_model import SC_Model
from inference.rag_model import RAG_Model
from inference.one_hop_kg_model import OneHopKG_Model
from inference.one_hop_kg_rag_model import OneHopKG_RAG_Model
from inference.tog_model import ToG_Model
from inference.pog_model import PoG_Model
from inference.our_model import Our_Model
from inference.our_minus_global_search_model import OurMinusGlobalSearch_Model
from inference.our_minus_route_model import OurMinusRoute_Model

MODEL_MAP = {
    "io": IO_Model,
    "cot": CoT_Model,
    "sc": SC_Model,
    "rag": RAG_Model,
    "one-hop-kg": OneHopKG_Model,
    "one-hop-kg-rag": OneHopKG_RAG_Model,
    "tog": ToG_Model,
    "pog": PoG_Model,
    "our": Our_Model,
    "our_minus_global_search": OurMinusGlobalSearch_Model,
    "our_minus_route": OurMinusRoute_Model,
}
