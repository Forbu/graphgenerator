pip3 install networkx poetry pytest
pip3 install -U 'tensorboardX'

# NB_LAYER
NB_LAYER=$1

# hidden_size
HIDDEN_SIZE=$2

# nb_graphs
NB_GRAPHS=$3
# batch_size
BATCH_SIZE=$4

cd /app/data/scripts_training/ && python3 train_diffusion.py --nb_layer $NB_LAYER --hidden_dim $HIDDEN_SIZE --nb_graphs $NB_GRAPHS --batch_size $BATCH_SIZE
