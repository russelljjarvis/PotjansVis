

def plot_conn_sankey(conv_kernel,ncells):
    import plotly.graph_objects as go
    import pandas as pd
    edges_df=[]
    for y,i in enumerate(conv_kernel.weights.init):
        for x,j in enumerate(i):
            if j!=0.0:
                edges={'src':y,'tgt':int(x+ncells),'weight':np.abs(j)}
                edges_df.append(edges)
    edges_df = pd.DataFrame(edges_df)
    edges_df = edges_df[0:15]


    labels = edges_df.index
    labels = list(labels)
    temp = list(edges_df["tgt"].values)

    labels.extend(temp)

    data = dict(
        type="sankey",
        node=dict(
            hoverinfo="all",
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=edges_df["src"], target=edges_df["tgt"], value=edges_df["weight"]
        ),
    )

    layout = dict(title='Layered Network Connectivity', font=dict(size=10))

    fig = go.Figure(data=[data], layout=layout)
    fig.show()


def compute_cv(spikes_population):
    '''
    Compute coefficient of variation on a whole population of spike trains.
    '''
    train_of_trains = []
    for spike_train in spikes_population.T:
        train_of_trains.extend(spike_train)
    return elephant.statistics.cv(train_of_trains, axis=0, nan_policy='propagate')

#@st.cache
def generate_sankey_figure(
    nodes_list, edges_df, title = "neural connectivity"
):
    import plotly.graph_objects as go


    edges_df["src"] = edges_df["src"].apply(lambda x: nodes_list.index(x))
    edges_df["tgt"] = edges_df["tgt"].apply(lambda x: nodes_list.index(x))
    # creating the sankey diagram
    data = dict(
        type="sankey",
        node=dict(
            hoverinfo="all",
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_list,
        ),
        link=dict(
            source=edges_df["src"], target=edges_df["tgt"], value=100*edges_df["weight"]
        ),
    )

    layout = dict(title='Layered Network Connectivity', font=dict(size=50))
    fig = go.Figure(data=[data], layout=layout)
    fig.update_layout(
        autosize=False,
        width=1100,
        height=1100
    )

    return fig

def cached_chord(first):
    H = first.to_undirected()
    centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

    # centrality = nx.betweenness_centrality(H)#, endpoints=True)
    df = pd.DataFrame([centrality])
    df = df.T
    df.sort_values(0, axis=0, ascending=False, inplace=True)
    bc = df
    bc.rename(columns={0: "centrality value"}, inplace=True)

    temp = pd.DataFrame(first.nodes)
    nodes = hv.Dataset(temp[0])

    links = copy.copy(adj_mat)
    links.rename(
        columns={"weight": "value", "src": "source", "tgt": "target"}, inplace=True
    )
    links = links[links["value"] != 0]

    Nodes_ = set(
        links["source"].unique().tolist() + links["target"].unique().tolist()
    )
    Nodes = {node: i for i, node in enumerate(Nodes_)}

    df_links = links.replace({"source": Nodes, "target": Nodes})
    for k in Nodes.keys():
        if k not in color_code_0.keys():
            color_code_0[k] = "Unknown"

    df_nodes = pd.DataFrame(
        {
            "index": [idx for idx in Nodes.values()],
            "name": [name for name in Nodes.keys()],
            "colors": [color_code_0[k] for k in Nodes.keys()],
        }
    )
    dic_to_sort = {}
    for i, kk in enumerate(df_nodes["name"]):
        dic_to_sort[i] = color_code_0[k]

    t = pd.Series(dic_to_sort)
    df_nodes["sort"] = t  # pd.Series(df_links.source)
    df_nodes.sort_values(by=["sort"], inplace=True)

    dic_to_sort = {}
    for i, kk in enumerate(df_links["source"]):
        k = df_nodes.loc[kk, "name"]
        # st.text(k)
        if k not in color_code_0.keys():
            color_code_0[k] = "Unknown"
        df_nodes.loc[kk, "colors"] = color_code_0[k]
        dic_to_sort[i] = color_code_0[k]

    pd.set_option("display.max_columns", 11)
    hv.extension("bokeh")
    hv.output(size=200)
    t = pd.Series(dic_to_sort)
    df_links["sort"] = t  # pd.Series(df_links.source)
    df_links.sort_values(by=["sort"], inplace=True)
    # df_links['colors'] = None
    categories = np.unique(df_links["sort"])
    colors = np.linspace(0, 1, len(categories))
    colordicth = dict(zip(categories, colors))

    df_links["Color"] = df_links["sort"].apply(lambda x: float(colordicth[x]))
    colors = df_links["Color"].values
    nodes = hv.Dataset(df_nodes, "index")
    df_links["index"] = df_links["Color"]
    chord = hv.Chord(
        (df_links, nodes)
    )  # .opts.Chord(cmap='Category20', edge_color=dim('source').astype(str), node_color=dim('index').astype(str))
    chord.opts(
        opts.Chord(
            cmap="Category20",
            edge_cmap="Category20",
            edge_color=dim("sort").str(),
            width=350,
            height=350,
            labels="Color",
        )
    )

    hv.save(chord, "chord2.html", backend="bokeh")
