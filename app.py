import streamlit as st
import potjans_as_network as pan
import pandas as pd
from net_viz import plot_conn_sankey, generate_sankey_figure
from net_viz import *
import network_params_pynn
import numpy as np
import elephant
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import networkx as nx
import pyNN.neuron as sim
sim.setup()
sim.end()
def nx_chunk(graph, chunk_size):
    """
    Chunk a graph into subgraphs with the specified minimum chunk size.

    Inspired by Lukes algorithm.
    """

    # convert to a tree;
    # a balanced spanning tree would be preferable to a minimum spanning tree,
    # but networkx does not seem to have an implementation of that
    tree = nx.minimum_spanning_tree(graph)

    # select a root that is maximally far away from all leaves
    leaves = [node for node, degree in tree.degree() if degree == 1]
    minimum_distance_to_leaf = {node : tree.size() for node in tree.nodes()}
    for leaf in leaves:
        distances = nx.single_source_shortest_path_length(tree, leaf)
        for node, distance in distances.items():
            if distance < minimum_distance_to_leaf[node]:
                minimum_distance_to_leaf[node] = distance
    root = max(minimum_distance_to_leaf, key=minimum_distance_to_leaf.get)

    # make the tree directed and compute the total descendants of each node
    tree = nx.dfs_tree(tree, root)
    total_descendants = get_total_descendants(tree)

    # prune chunks, starting from the leaves
    chunks = []
    max_descendants = np.max(list(total_descendants.values()))
    while (max_descendants + 1 > chunk_size) & (tree.size() >= 2 * chunk_size):
        for node in list(nx.topological_sort(tree))[::-1]: # i.e. from leaf to root
            if (total_descendants[node] + 1) >= chunk_size:
                chunk = list(nx.descendants(tree, node))
                chunk.append(node)
                chunks.append(chunk)

                # update relevant data structures
                tree.remove_nodes_from(chunk)
                total_descendants = get_total_descendants(tree)
                max_descendants = np.max(list(total_descendants.values()))

                break

    # handle remainder
    chunks.append(list(tree.nodes()))

    return chunks


def get_total_descendants(dag):
    return {node : len(nx.descendants(dag, node)) for node in dag.nodes()}



def main():
    genre = st.sidebar.radio(
        "Choose Graph Layout/Option:",
        (
            "Sankey",
            "Force Directed",
            "Community Based Load Balance",
        ),
    )

    if genre == "Sankey":
        my_expander = st.expander("Explanation of Sankey")
        my_expander.markdown(
            """Communities in the graph on the left are not IRG 1-3, but instead communities found by blind network analysis. It's appropritate to use a different color code for the four inferred communities. \
        For contrast in the graph on the right, machine driven community detection clusters persist, but now nodes (dots) are color coded IRG-1-3 \n \
        This suggests that the formal memberships eg. \"IRG 1\" does not determine the machine generated communities. In otherwords spontaneuosly emerging community groups may be significantly different to formal group assignments.
        The stochastic community detection algorithm uses a differently seeded random number generator every time so the graph appears differently each time the function is called.
        The algorithm is called Louvain community detection. The Louvain Community algorithm detects 5 communities, but only 2 communities with membership >=3. A grey filled convex hull is drawn around each of the two larger communities.
        """
        )


        #G = pan.G

        ei_ratio = st.slider('Desired Weight ratio', 0.0, 1.0, 0.01)

        ratio,G,weights = pan.set_weight_ratio(pan.enum_node_name,pan.edges,known_ratio=ei_ratio)
        st.write("Excit Inhib", ei_ratio,ratio)

        list_of_dicts=[]
        cnt=0
        for edge in G.edges:
            list_of_dicts.append({'src':edge[0],'tgt':edge[1],'weight':weights[cnt]})
            cnt+=1
        df = pd.DataFrame(list_of_dicts)
        fig = generate_sankey_figure(list(G.nodes),df)
        st.write(fig)

    if genre == "Force Directed":
        #cell_count = st.slider('Desired Cell Count Scale', 0.5, 10.0, 0.1)

        ei_ratio = st.slider('Desired Weight ratio', 0.0, 1.0, 0.01)
        ratio,G,weights = pan.set_weight_ratio(pan.enum_node_name,pan.edges,known_ratio=ei_ratio)
        st.write("Excit Inhib", ei_ratio,ratio)

        nt = pan.interactive_population(pan.node_name,G,weights,pan.cd)
        nt.save_graph("population.html")
        HtmlFile = open("population.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        components.html(source_code, height=800, width=800)  # ,use_column_width=True)



    #def dontdo():
    if genre == "Community Based Load Balance":
        my_expander = st.expander("Explanation of Community Partitions")
        my_expander.markdown(
            """Communities in the graph on the left are not IRG 1-3, but instead communities found by blind network analysis. It's appropritate to use a different color code for the four inferred communities. \
        For contrast in the graph on the right, machine driven community detection clusters persist, but now nodes (dots) are color coded IRG-1-3 \n \
        This suggests that the formal memberships eg. \"IRG 1\" does not determine the machine generated communities. In otherwords spontaneuosly emerging community groups may be significantly different to formal group assignments.
        The stochastic community detection algorithm uses a differently seeded random number generator every time so the graph appears differently each time the function is called.
        The algorithm is called Louvain community detection. The Louvain Community algorithm detects 5 communities, but only 2 communities with membership >=3. A grey filled convex hull is drawn around each of the two larger communities.
        """
        )
        ei_ratio = st.slider('Desired Weight ratio', 0.0, 1.0, 0.01)
        ratio,g,weights = pan.set_weight_ratio(pan.enum_node_name,pan.edges,known_ratio=ei_ratio)
        st.write("Excit Inhib", ei_ratio,ratio)

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))

        chunks = nx_chunk(g, 4)
        #node_to_color = dict()
        print(len(chunks))
        for ii, chunk in enumerate(chunks):
            for node in chunk:
                node_to_color[node] = ii
        nx.draw(g, node_color=[node_to_color[node] for node in g.nodes()], cmap='tab20', ax=axes)
        st.pyplot(fig, use_column_width=True)

if __name__ == "__main__":
    main()
