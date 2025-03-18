pip3 install networkx poetry pytest heavyball lightning matplotlib x-transformers torch_geometric
pip3 install -U 'tensorboardX'

apt update
apt-get -y install build-essential

cd /app/data/
python3 -m scripts_training.test_g2pt_llada_pp
