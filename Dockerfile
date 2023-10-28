FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# install networkx for graph generation
RUN pip3 install networkx poetry pytest
RUN pip3 install -U 'tensorboardX'
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN pip3 install torch_geometric