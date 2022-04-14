
apt-get update
wget https://github.com/lava-nc/lava/releases/download/v0.2.0/lava-nc-0.2.0.tar.gz
pip install lava-nc-0.1.0.tar.gz

#git clone https://github.com/lava-nc/lava
#cd lava

python -m pip install numpy --upgrade --ignore-installed
python -m pip install cython --upgrade --ignore-installed

python -m pip install -r requirements.txt
#conda install -c pyviz scikit-image# bokeh graphviz_layout
#python -m conda install -c pyvis bokeh seaborn scikit-image # dash_bio
#python -m pip install bokeh# holoviews==1.14.1 seaborn
python -m pip install pyvis cython scikit-image #dash_bio dask
python -m pip install plotly tabulate # hiveplotlib hiveplot pygraphviz#==2.0.0#2.2
#python -m pip install streamlit --upgrade --ignore-installed
#python -m pip install pygraphviz
# streamlit-agraph
#python -m pip install python-igraph

#$(which python) -m pip install git+https://github.com/taynaud/python-louvain.git@networkx2
#python -m pip install git+https://github.com/taynaud/python-louvain.git@networkx2


python -m pip install python-louvain #git+https://github.com/taynaud/python-louvain.git@networkx2
#git clone https://github.com/taynaud/python-louvain.git@networkx2; cd networkx2; python setup.py install; cd -
#python make_serial_plots0.py
#python make_serial_plots1.py
#python -c "from holoviews.operation.datashader import datashade"
#python -m pip install git+https://github.com/pyviz/holoviews.git
#git clone https://github.com/pyviz/holoviews.git
#cd holoviews; pip install -e .; cd ..;
#python -m pip install pygraphviz


mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"russelljarvis@protonmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
