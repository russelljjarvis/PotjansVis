


from pyvis.network import Network
from dicthash import dicthash
import network_params_pynn
import numpy as np
import networkx as nx
nodes = network_params_pynn.N_full.keys()
def interactive_population(sizes, popg, weights,color_dict):


    node_color = [color_dict[n] for n in popg]
    edge_colors = []
    for e in popg.edges:
        edge_colors.append(color_dict[e[0]])

    temp = list([s * 1000 for s in sizes.values()])
    widths = []
    edge_list = []
    edge_colors = []
    for e in popg.edges:
        edge_list.append((e[0], e[1]))
    labels = {}
    for node in popg.nodes():
        # set the node name as the key and the label as its value
        labels[node] = node
    nt = Network("700px", "700px",directed=True,notebook=True)  # ,layout=physics_layouts)

    nt.barnes_hut()
    for node in popg.nodes:
        nt.add_node(node,label=node) # node id = 1 and label = Node 1
    nt.set_edge_smooth('continuous')
    cnt=0
    for e in popg.edges:
        src = e[0]
        dst = e[1]
        src = str(src)
        dst = str(dst)
        ee = popg.get_edge_data(e[0], e[1])
        nt.add_edge(src, dst, width=weights[cnt]*100)
        cnt+=1

    for i,node in enumerate(nt.nodes):
        node["size"] = sizes[node["id"]]/700 #* 1025
        node["color"] = cd[node["id"]]

    for node in nt.nodes:
        node["title"] = (
            "<br> {0}'s' group size is: {1}<br>".format(node["id"],sizes[node["id"]])
        )

    nt.save_graph("population.html")
    return nt#, weights

network_params_pynn.N_full
node_name = {}
cnt = 0
enum_node_name={}
color_dict_ = {
    "1": "blue",
    "2": "red",
    "3": "green",
    "0": "purple",
    "4": "blue",
    "5": "red",
    "6": "green",
    "7": "purple",
    "8": "blue",
    "9": "red",
    }

cd = {}
for (k,v) in network_params_pynn.N_full.items():
    node_name[str(k)+str("E")] = v["E"]
    enum_node_name[cnt] = str(k)+str("E")
    cd[str(k)+str("E")] = color_dict_[str(cnt)]
    cnt += 1
    node_name[str(k)+str("I")] = v["I"]
    enum_node_name[cnt] = str(k)+str("I")
    cd[str(k)+str("I")] = color_dict_[str(cnt)]
    cnt += 1

#node_name;
#enum_node_name;
edges = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
             [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
             [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
             [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
             [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]




def set_weight_ratio(enum_node_name,edges,known_ratio):
    weights = []
    Iwtotal=[]
    Ewtotal=[]
    G=nx.Graph()

    for k,v in enum_node_name.items():
        for k_,v_ in enum_node_name.items():
            weight=edges[k_][k]

            if "I" in v_:
                weight = weight*(1.0+known_ratio)
                print("I",weight)
                G.add_edge(v,v_,w=weight)
                Iwtotal.append(weight)
            if "E" in v_:
                weight = weight*(1.0-known_ratio)
                print("E",weight)

                G.add_edge(v,v_,w=weight)
                Ewtotal.append(weight)

            weights.append(weight)

    ratio = sum(Ewtotal)/sum(Iwtotal)
    return ratio,G,weights
#G.ratio
