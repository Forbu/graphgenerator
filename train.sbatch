#!/bin/bash -l
# SBATCH --job-name "throughput_map_generation"


# Set up the environment
CONTAINER_IMAGE=registry.gitlab.tech.orange/radio-planning-software/starlight/stardust:prediction
CONTAINER_NAME=stardust_prediction
CONTAINER_MOUNTS=/opt/marcel-c3/workdir/mccf3797/graphgenerator:/app/data

# retrieve the nb_cpu argument
NB_CPU=$1

# retrieve the nb_gpu argument
NB_GPU=$2

# NB_LAYER
NB_LAYER=$3

# hidden_size
HIDDEN_SIZE=$4

# nb_graphs
NB_GRAPHS=$5

# batch_size
BATCH_SIZE=$6

echo "Launching container $CONTAINER_NAME with image $CONTAINER_IMAGE"
echo "Mounting $CONTAINER_MOUNTS"

# take in consideration the number of GPUs and the time limit (and also the log file name)
srun -c $NB_CPU --gpus=$NB_GPU --container-image=$CONTAINER_IMAGE --container-name=$CONTAINER_NAME --container-mounts=$CONTAINER_MOUNTS \
     bash /app/data/launch.sh $NB_LAYER $HIDDEN_SIZE $NB_GRAPHS $BATCH_SIZE

