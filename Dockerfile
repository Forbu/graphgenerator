FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# install networkx for graph generation
RUN pip3 install networkx poetry pytest
