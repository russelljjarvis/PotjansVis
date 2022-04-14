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
import pyNN.brian2 as sim
#sim.end()
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.brian2 import *




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
        st.text("Excit Inhib", ei_ratio,ratio)

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
        st.text("Excit Inhib", ei_ratio,ratio)

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
        st.text("Excit Inhib", ei_ratio,ratio)

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))

        chunks = nx_chunk(g, 4)
        #node_to_color = dict()
        st.text(len(chunks))
        for ii, chunk in enumerate(chunks):
            for node in chunk:
                node_to_color[node] = ii
        nx.draw(g, node_color=[node_to_color[node] for node in g.nodes()], cmap='tab20', ax=axes)
        st.pyplot(fig, use_column_width=True)



    #timer = Timer()

    # === Define parameters ========================================================

    downscale   = 50      # scale number of neurons down by this factor
                          # scale synaptic weights up by this factor to
                          # obtain similar dynamics independent of size
    order       = 500  # determines size of network:
                          # 4*order excitatory neurons
                          # 1*order inhibitory neurons
    Nrec        = 10      # number of neurons to record from, per population
    epsilon     = 0.1     # connectivity: proportion of neurons each neuron projects to

    # Parameters determining model dynamics, cf Brunel (2000), Figs 7, 8 and Table 1
    # here: Case C, asynchronous irregular firing, ~35 Hz
    eta         = 2.0     # rel rate of external input
    g           = 5.0     # rel strength of inhibitory synapses
    J           = 0.1     # synaptic weight [mV]
    delay       = 1.5     # synaptic delay, all connections [ms]

    # single neuron parameters
    tauMem      = 20.0    # neuron membrane time constant [ms]
    tauSyn      = 0.1     # synaptic time constant [ms]
    tauRef      = 2.0     # refractory time [ms]
    U0          = 0.0     # resting potential [mV]
    theta       = 20.0    # threshold

    # simulation-related parameters
    simtime     = 1.0   # simulation time [ms]
    dt          = 0.1     # simulation step length [ms]

    # seed for random generator used when building connections
    connectseed = 12345789
    use_RandomArray = True  # use Python rng rather than NEST rng

    # seed for random generator(s) used during simulation
    kernelseed  = 43210987

    # === Calculate derived parameters =============================================

    # scaling: compute effective order and synaptic strength
    order_eff = int(float(order)/downscale)
    J_eff     = J*downscale

    # compute neuron numbers
    NE = int(4*order_eff)  # number of excitatory neurons
    NI = int(1*order_eff)  # number of inhibitory neurons
    N  = NI + NE           # total number of neurons

    # compute synapse numbers
    CE   = int(epsilon*NE)  # number of excitatory synapses on neuron
    CI   = int(epsilon*NI)  # number of inhibitory synapses on neuron
    C    = CE + CI          # total number of internal synapses per n.
    Cext = CE               # number of external synapses on neuron

    # synaptic weights, scaled for alpha functions, such that
    # for constant membrane potential, charge J would be deposited
    fudge = 0.00041363506632638  # ensures dV = J at V=0

    # excitatory weight: JE = J_eff / tauSyn * fudge
    JE = (J_eff/tauSyn)*fudge

    # inhibitory weight: JI = - g * JE
    JI = -g*JE

    # threshold, external, and Poisson generator rates:
    nu_thresh = theta/(J_eff*CE*tauMem)
    nu_ext    = eta*nu_thresh     # external rate per synapse
    p_rate    = 1000*nu_ext*Cext  # external input rate per neuron (Hz)

    # number of synapses---just so we know
    Nsyn = (C+1)*N + 2*Nrec  # number of neurons * (internal synapses + 1 synapse from PoissonGenerator) + 2synapses" to spike detectors

    # put cell parameters into a dict
    cell_params = {'tau_m'      : tauMem,
                   'tau_syn_E'  : tauSyn,
                   'tau_syn_I'  : tauSyn,
                   'tau_refrac' : tauRef,
                   'v_rest'     : U0,
                   'v_reset'    : U0,
                   'v_thresh'   : theta,
                   'cm'         : 0.001}     # (nF)

    # === Build the network ========================================================

    # clear all existing network elements and set resolution and limits on delays.
    # For NEST, limits must be set BEFORE connecting any elements

    #extra = {'threads' : 2}
    extra = {}

    rank = setup(timestep=dt, max_delay=delay, **extra)

    st.text("%d Setting up random number generator" % rank)
    rng = NumpyRNG(kernelseed, parallel_safe=True)

    st.text("%d Creating excitatory population with %d neurons." % (rank, NE))
    celltype = IF_curr_alpha(**cell_params)
    E_net = Population(NE, celltype, label="E_net")

    st.text("%d Creating inhibitory population with %d neurons." % (rank, NI))
    I_net = Population(NI, celltype, label="I_net")

    st.text("%d Initialising membrane potential to random values between %g mV and %g mV." % (rank, U0, theta))
    uniformDistr = RandomDistribution('uniform', low=U0, high=theta, rng=rng)
    E_net.initialize(v=uniformDistr)
    I_net.initialize(v=uniformDistr)

    st.text("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_type = SpikeSourcePoisson(rate=p_rate)
    expoisson = Population(NE, source_type, label="expoisson")

    st.text("%d Creating inhibitory Poisson generator with the same rate." % rank)
    inpoisson = Population(NI, source_type, label="inpoisson")

    # Record spikes
    st.text("%d Setting up recording in excitatory population." % rank)
    E_net.sample(Nrec).record('spikes')
    #E_net[0:2].record('v')

    st.text("%d Setting up recording in inhibitory population." % rank)
    I_net.sample(Nrec).record('spikes')
    #I_net[0:2].record('v')
    connector = FixedProbabilityConnector(epsilon, rng=rng)
    E_syn = StaticSynapse(weight=JE, delay=delay)
    I_syn = StaticSynapse(weight=JI, delay=delay)
    ext_Connector = OneToOneConnector()
    ext_syn = StaticSynapse(weight=JE, delay=dt)

    st.text("%d Connecting excitatory population with connection probability %g, weight %g nA and delay %g ms." % (rank, epsilon, JE, delay))
    E_to_E = Projection(E_net, E_net, connector, E_syn, receptor_type="excitatory")
    st.text("E --> E\t\t", len(E_to_E), "connections")
    I_to_E = Projection(I_net, E_net, connector, I_syn, receptor_type="inhibitory")
    st.text("I --> E\t\t", len(I_to_E), "connections")
    input_to_E = Projection(expoisson, E_net, ext_Connector, ext_syn, receptor_type="excitatory")
    st.text("input --> E\t", len(input_to_E), "connections")

    st.text("%d Connecting inhibitory population with connection probability %g, weight %g nA and delay %g ms." % (rank, epsilon, JI, delay))
    E_to_I = Projection(E_net, I_net, connector, E_syn, receptor_type="excitatory")
    st.text("E --> I\t\t", len(E_to_I), "connections")
    I_to_I = Projection(I_net, I_net, connector, I_syn, receptor_type="inhibitory")
    st.text("I --> I\t\t", len(I_to_I), "connections")
    input_to_I = Projection(inpoisson, I_net, ext_Connector, ext_syn, receptor_type="excitatory")
    st.text("input --> I\t", len(input_to_I), "connections")

    # read out time used for building
    buildCPUTime = timer.elapsedTime()
    # === Run simulation ===========================================================

    # run, measure computer time
    timer.start()  # start timer on construction
    st.text("%d Running simulation for %g ms." % (rank, simtime))
    run(simtime)
    simCPUTime = timer.elapsedTime()
    st.text(simCPUTime)
    # write data to file
    st.text("%d Writing data to file." % rank)
    (E_net + I_net).write_data("Results/brunel_np%d_%s.pkl" % (np, simulator_name))

    E_rate = E_net.mean_spike_count()*1000.0/simtime
    I_rate = I_net.mean_spike_count()*1000.0/simtime

    # write a short report
    nst.text("\n--- Brunel Network Simulation ---")
    nst.text("Nodes              : %d" % np)
    nst.text("Number of Neurons  : %d" % N)
    nst.text("Number of Synapses : %d" % Nsyn)
    nst.text("Input firing rate  : %g" % p_rate)
    nst.text("Excitatory weight  : %g" % JE)
    nst.text("Inhibitory weight  : %g" % JI)
    nst.text("Excitatory rate    : %g Hz" % E_rate)
    nst.text("Inhibitory rate    : %g Hz" % I_rate)
    nst.text("Build time         : %g s" % buildCPUTime)
    st.text("Simulation time    : %g s" % simCPUTime)

if __name__ == "__main__":
    main()
