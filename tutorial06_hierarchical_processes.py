


from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import elephant

class Dense(AbstractProcess):
    """Dense connections between neurons.
    Realizes the following abstract behavior:
    a_out = W * s_in
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))


# #### Create a Python Dense connection _ProcessModel_ implementing the Loihi Sync Protocol and requiring a CPU compute resource

# In[2]:


import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
#from lava.proc.dense.process import Dense


@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float)
    weights: np.ndarray = LavaPyType(np.ndarray, np.float)

    def run_spk(self):
        s_in = self.s_in.recv()
        a_out = self.weights[:, s_in].sum(axis=1)
        self.a_out.send(a_out)


# #### Create a LIF neuron _Process_

# In[3]:


from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class LIF(AbstractProcess):
    """Leaky-Integrate-and-Fire (LIF) neural Process.
    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike
    Parameters
    ----------
    du: Inverse of decay time-constant for current decay.
    dv: Inverse of decay time-constant for voltage decay.
    bias: Neuron bias.
    vth: Neuron threshold voltage, exceeding which, the neuron will spike.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias = kwargs.pop("bias", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias = Var(shape=shape, init=bias)
        self.vth = Var(shape=(1,), init=vth)
        #self.spikes = Var(shape=shape, init=0)


# #### Create a Python LIF neuron _ProcessModel_ implementing the Loihi Sync Protocol and requiring a CPU compute resource

# In[4]:


import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
#from lava.proc.lif.process import LIF


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
class PyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.float)
    v: np.ndarray = LavaPyType(np.ndarray, np.float)
    bias: np.ndarray = LavaPyType(np.ndarray, np.float)
    du: float = LavaPyType(float, np.float)
    dv: float = LavaPyType(float, np.float)
    vth: float = LavaPyType(float, np.float)
    spikes: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += a_in_data
        self.v[:] = self.v * (1 - self.dv) + self.u + self.bias
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        #self.spikes = s_out
        self.s_out.send(s_out)


# ## Create a DenseLayer Hierarchical _Process_ that encompasses Dense and LIF _Process_ behavior

# Now we create a DenseLayer _Hierarchical Process_ combining LIF neural _Processes_ and Dense connection _Processes_. Our _Hierarchical Process_ contains all of the variables (`u`, `v`, `bias`, `du`, `dv`, `vth`, and `s_out`) native to the LIF _Process_ plus the `weights` variable native to the Dense _Process_. The InPort to our _Hierarchical Process_ is `s_in`, which represents the spike inputs to our Dense synaptic connections. These Dense connections synapse onto a population of LIF neurons. The OutPort of our _Hierarchical Process_ is `s_out`, which represents the spikes output by the layer of LIF neurons.

# In[5]:


class DenseLayer(AbstractProcess):
    """Combines Dense and LIF Processes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias = kwargs.pop("bias", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)
        weights = kwargs.pop("weights", 0)

        self.s_in = InPort(shape=(shape[1],))
        #output of Dense synaptic connections is only used internally
        #self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=weights)
        #input to LIF population from Dense synaptic connections is only used internally
        #self.a_in = InPort(shape=(shape[0],))
        self.s_out = OutPort(shape=(shape[0],))
        self.u = Var(shape=(shape[0],), init=0)
        self.v = Var(shape=(shape[0],), init=0)
        self.bias = Var(shape=(shape[0],), init=bias)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.vth = Var(shape=(1,), init=vth)
        #self.spikes = Var(shape=(shape[0],), init=0)


# ## Create a _SubProcessModel_ that implements the DenseLayer _Process_ using Dense and LIF child _Processes_

# Now we will create the _SubProcessModel_ that implements our DenseLayer _Process_. This inherits from the _AbstractSubProcessModel_ class. Recall that _SubProcessModels_ also inherit the compute resource requirements from the _ProcessModels_ of their child _Processes_. In this example, we will use the LIF and Dense _ProcessModels_ requiring a CPU compute resource that were defined earlier in the tutorial,  and `SubDenseLayerModel` will therefore implicitly require the CPU compute resource.
#
# The `__init__()` constructor of `SubDenseLayerModel` builds the sub _Process_ structure of the `DenseLayer` _Process_. The `DenseLayer` _Process_ gets passed to the `__init__()` method via the `proc` attribute. The `__init__()` constructor first instantiates the child LIF and Dense _Processes_. Initial conditions of the `DenseLayer` _Process_, which are required to instantiate the child LIF and Dense _Processes_, are accessed through `proc.init_args`.
#
# We then `connect()` the in-port of the Dense child _Process_ to the in-port of the `DenseLayer` parent _Process_ and the out-port of the LIF child _Process_ to the out-port of the `DenseLayer` parent _Process_. Note that ports of the `DenseLayer` parent process are accessed using `proc.in_ports` or `proc.out_ports`, while ports of a child _Process_ like LIF are accessed using `self.lif.in_ports` and `self.lif.out_ports`. Our _ProcessModel_ also internally `connect()`s the out-port of the Dense connection child _Process_ to the in-port of the LIF neural child _Process_.
#
# The `alias()` method exposes the variables of the LIF and Dense child _Processes_ to the `DenseLayer` parent _Process_. Note that the variables of the `DenseLayer` parent _Process_ are accessed using `proc.vars`, while the variables of a child _Process_ like LIF are accessed using `self.lif.vars`. Note that unlike a _LeafProcessModel_, a _SubProcessModel_ does not require variables to be initialized with a specified data type or precision. This is because the data types and precisions of all `DenseLayer` _Process_ variables (`proc.vars`) are determined by the particular _ProcessModels_ chosen by the Run Configuration to implement the LIF and Dense child _Processes_. This allows the same _SubProcessModel_ to be used flexibly across multiple languages and compute resources when the child _Processes_ have multiple _ProcessModel_ implementations. _SubProcessModels_ thus enable the composition of complex applications agnostic of platform-specific implementations. In this example, we will implement the LIF and Dense _Processes_ with the _PyLoihiProcessModels_ defined earlier in the tutorial, so the `DenseLayer` variables aliased from LIF and Dense implicity have type `LavaPyType` and precisions as specified in `PyLifModel` and `PyDenseModel`.

# In[6]:


import numpy as np

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.decorator import implements

@implements(proc=DenseLayer, protocol=LoihiProtocol)
class SubDenseLayerModel(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        # Instantiate child processes
        #input shape is a 2D vec (shape of weight mat)
        shape = proc.init_args.get("shape",(1,1))
        weights = proc.init_args.get("weights",(1,1))
        bias = proc.init_args.get("bias",(1,1))
        vth = proc.init_args.get("vth",(1,1))
        #shape is a 2D vec (shape of weight mat)
        self.dense = Dense(shape=shape, weights=weights)
        #shape is a 1D vec
        self.lif = LIF(shape=(shape[0],),bias=bias,vth=vth)
        # connect Parent in port to child Dense in port
        proc.in_ports.s_in.connect(self.dense.in_ports.s_in)
        # connect Dense Proc out port to LIF Proc in port
        self.dense.out_ports.a_out.connect(self.lif.in_ports.a_in)
        # connect child LIF out port to parent out port
        self.lif.out_ports.s_out.connect(proc.out_ports.s_out)

        proc.vars.u.alias(self.lif.vars.u)
        proc.vars.v.alias(self.lif.vars.v)
        proc.vars.bias.alias(self.lif.vars.bias)
        proc.vars.du.alias(self.lif.vars.du)
        proc.vars.dv.alias(self.lif.vars.dv)
        proc.vars.vth.alias(self.lif.vars.vth)
        proc.vars.weights.alias(self.dense.vars.weights)
        #proc.vars.spikes.alias(self.lif.vars.spikes)


# ## Run the DenseLayer _Process_

# #### Run Connected DenseLayer _Processes_

# In[12]:
'''
cached_chord(first)
HtmlFile2 = open("chord2.html", "r", encoding="utf-8")
source_code2 = HtmlFile2.read()
components.html(source_code2, height=750, width=750)
'''
'''
def make_sankey_chart(df, transport_types):

    import plotly.graph_objects as go

    encode = OrderedDict()
    transport_types = list(transport_types)
    for i, name in enumerate(transport_types):
        encode[name] = 4 + i

    less_five_src = df[df["One-Way Daily Commute Distance (km)"] < 5.0].index
    less_five_src = [1.0 for i in range(0, len(less_five_src))]
    less_five_tgt = df[df["One-Way Daily Commute Distance (km)"] < 5.0][
        "Main Transport Mode"
    ]
    less_five_tgt = encode_list(less_five_tgt, encode)
    df_filtered = df[df["One-Way Daily Commute Distance (km)"] >= 5.0]
    df_filtered = df_filtered[df_filtered["One-Way Daily Commute Distance (km)"] < 10.0]

    less_ten_src = (
        df_filtered.index
    )
    less_ten_src = [2.0 for i in range(0, len(less_ten_src))]
    less_ten_tgt = df_filtered["Main Transport Mode"]
    less_ten_tgt = encode_list(less_ten_tgt, encode)

    greater_ten_src = df[df["One-Way Daily Commute Distance (km)"] >= 10.0].index
    greater_ten_src = [3.0 for i in range(0, len(greater_ten_src))]

    greater_ten_tgt = df[df["One-Way Daily Commute Distance (km)"] >= 10.0][
        "Main Transport Mode"
    ]
    greater_ten_tgt = encode_list(greater_ten_tgt, encode)

    srcs = []
    srcs.extend(less_five_src)
    srcs.extend(less_ten_src)
    srcs.extend(greater_ten_src)
    tgts = []
    tgts.extend(less_five_tgt)
    tgts.extend(less_ten_tgt)
    tgts.extend(greater_ten_tgt)
    assert len(srcs) == len(tgts)
    labels = ["less than 5km", "between 5km and 10km", "greater than 10km"]
    labels.extend(transport_types)
    labels.insert(0, "less than 5km")
    assert len(srcs) == len(tgts)
    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]
    encode_list(transport_types, encode)

    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat=".0f",
                #valuesuffix="TWh",
                # Define nodes


                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors,
                ),
                # Add links
                link=dict(
                    source=srcs,
                    target=tgts,
                    value=srcs,
                ),
            )
        ]
    )

    #fig.update_layout(title_text="", font_size=10)


    fig.update_layout(
    hovermode = 'x',
    font=dict(size = 10, color = 'white'),
    plot_bgcolor='black',
    paper_bgcolor='black')
    return fig
'''
#rcfg = Loihi1SimCfg(select_tag='floating_pt', select_sub_proc_model=True)
'''
for t in range(9):
    #running layer 1 runs all connected layers (layer 0)
    layer1.run(condition=RunSteps(num_steps=1),run_cfg=rcfg)
    print('t: ',t)
    print('Layer 0 v: ', layer0.v.get())
    print('Layer 1 u: ', layer1.u.get())
    print('Layer 1 v: ', layer1.v.get())
    #print('Layer 1 spikes: ', layer1.spikes.get())
    print('\n ----- \n')
'''

# ## How to learn more?
#
# If you want to find out more about _SubProcessModels_, have a look at the [Lava documentation](https://lava-nc.org/) or dive into the [source code](https://github.com/lava-nc/lava/tree/main/src/lava/magma/core/model/sub/model.py).
#
# To receive regular updates on the latest developments and releases of the Lava Software Framework please subscribe to [our newsletter](http://eepurl.com/hJCyhb).
