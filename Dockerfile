FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip3 install -U 'tensorboardX'
RUN pip3 install lightning
RUN pip3 install x-transformers

# install networkx for graph generation
RUN pip3 install networkx poetry pytest matplotlib
