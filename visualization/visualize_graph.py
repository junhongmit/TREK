from pyvis.network import Network
import seaborn as sns
import colorsys
from matplotlib.colors import to_hex, to_rgb, to_rgba
import textwrap

from kg.kg_rep import *

def wrap_label(label: str, max_width: int = 20) -> str:
    words = label.split()
    lines = []
    current = ""
    for word in words:
        if len(current + " " + word) <= max_width:
            current += " " + word if current else word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)

def adjust_lightness(color, amount=1.0):
    """
    Adjust the lightness of a color.
    
    Parameters:
    - color: a matplotlib color string or RGB(A) tuple.
    - amount: a factor where <1 darkens, >1 lightens. 1 means no change.
    
    Returns:
    - A tuple representing the adjusted RGBA color.
    """
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * amount))  # keep lightness in [0, 1]
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
    a = to_rgba(color)[3]  # preserve alpha
    return (r_new, g_new, b_new, a)

def visualize_reasoning_graph(result: Dict, 
                              hierarchical: bool = False,
                              notebook: bool = False, 
                              output_path="reasoning_graph.html"):
    net = Network(height="750px", width="100%", directed=True, notebook=notebook)
    if hierarchical:
        net.set_options("""
        {
            "layout": {
                "hierarchical": {
                "enabled": true,
                "direction": "LR",
                "sortMethod": "hubsize",
                "levelSeparation": 500
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                "nodeDistance": 175
                },
                "solver": "hierarchicalRepulsion"
            }
        }
        """)
    else:
        net.force_atlas_2based()

    # Create a unique color for each entity type
    unique_types = list(set(ent.entity.type for ent in result['entities']))
    base_colors = sns.color_palette("husl", len(unique_types))
    type_to_base_color = {t: base_colors[i] for i, t in enumerate(unique_types)}

    # Add nodes
    for relevant_entity in result['entities']:
        entity = relevant_entity.entity
        
        label = wrap_label(f"({relevant_entity.step}) {entity.type}: {entity.name}")
        title = f"{label}\nStep: {relevant_entity.step}\nScore: {relevant_entity.score}\n{wrap_label(entity_to_text(entity), max_width=80)}"
        base_color = type_to_base_color.get(entity.type, (0.8, 0.8, 0.8))

        lightness = 0.9 + 0.15 * relevant_entity.step
        adjusted_color = adjust_lightness(base_color, amount=lightness)
        hex_color = to_hex(adjusted_color)

        font_size = 10 + 20 * relevant_entity.score  # Scaled by score
        net.add_node(
            entity.id, 
            label=label, 
            title=title, 
            color=hex_color, 
            shape="box",
            level=relevant_entity.step,
            font={"size": font_size}
        )

    # Add edges
    for relevant_relation in result['relations']:
        relation = relevant_relation.relation
        
        source = relation.source.id
        target = relation.target.id
        title = f"{relation.name}\nRelevance: {relevant_relation.score}\n{wrap_label(relation_to_text(relation), max_width=80)}"
        net.add_edge(
            source, 
            target, 
            label=relation.name, 
            title=title, 
            arrows="to", 
            value=relevant_relation.score  # scale and ensure min width
        )
    net.show(output_path, notebook=notebook)
    html = net.generate_html()
    
    query = result['query']
    custom_header = textwrap.dedent(f"""\
    <h3>Reasoning Graph for Query</h3>
    <p><b>Query:</b> {query.query}<br>
    <b>Steps:</b> {query.subqueries}<br>
    <b>Ans:</b> {result['ans']}</p>
    """)
    
    html_with_header = html.replace("<body>", f"<body>{custom_header}")
    
    with open("reasoning_graph.html", "w") as f:
        f.write(html_with_header)