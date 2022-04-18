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



def main():
    st.title("Adjustable Network Connectivity of Potjans et al Model")
    genre = st.sidebar.radio(
        "Choose Graph Layout/Option:",
        (
            "Sankey",
            "Force Directed",
        ),
    )

    if genre == "Sankey":

        el = st.sidebar.slider('Excitation Level', 0.0, 2.0, 1.0)
        il = st.sidebar.slider('Inhibition Level', 0.0, 2.0, 1.0)

        G,weights = pan.set_weight_ratio(pan.enum_node_name,pan.edges,el,il)#,enodes,innodes)

        list_of_dicts=[]
        cnt=0
        for edge in G.edges:
            list_of_dicts.append({'src':edge[0],'tgt':edge[1],'weight':weights[cnt]})
            cnt+=1
        df = pd.DataFrame(list_of_dicts)
        fig = generate_sankey_figure(list(G.nodes),df)#,node_size)
        st.write(fig)

    if genre == "Force Directed":
        #cell_count = st.slider('Desired Cell Count Scale', 0.5, 10.0, 0.1)

        el = st.sidebar.slider('Excitation Level', 0.0, 2.0, 1.0)
        il = st.sidebar.slider('Inhibition Level', 0.0, 2.0, 1.0)
        enodes = st.sidebar.slider('Excitation Population Size', 0.0, 10.0, 1.0)
        inodes = st.sidebar.slider('Inhibition Population Size', 0.0, 10.0, 1.0)
        G,weights = pan.set_weight_ratio(pan.enum_node_name,pan.edges,el,il)#,ennodes,innodes)
        d = dict(G.degree)
        node_size = {}
        for k,v_ in list(d.items()):
            if "E" in k:
                node_size[k] = v_ * enodes
            else:
                node_size[k] = v_ * inodes

        nt = pan.interactive_population(node_size,G,weights,pan.cd)

        #nt = pan.interactive_population(pan.node_name,G,weights,pan.cd)
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
        g,weights = pan.set_weight_ratio(pan.enum_node_name,pan.edges,known_ratio=ei_ratio)
        st.text("Excit Inhib", ei_ratio,ratio)

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))
        #nx.draw(g, node_color=[node_to_color[node] for node in g.nodes()], cmap='tab20', ax=axes)
        #st.pyplot(fig, use_column_width=True)


    #sys.path.append(system_params['pyNN_path'])

    #import network

    # create network
    #start_netw = time.time()
    #import verbatim_formal_pynn
    #n = verbatim_formal_pynn.Network(sim)
    #K_full = verbatim_formal_pynn.get_indegrees()
    #print(K_full)


if __name__ == "__main__":
    main()
