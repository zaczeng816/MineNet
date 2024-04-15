#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
# Purge modules
module purge

# Load Singularity container
singularity exec --nv \
  --overlay /scratch/wz1492/overlay-25GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
  /bin/bash -c "source /scratch/wz1492/env.sh;"

# Define hyperparameter combinations
models=("resnet50" "efficientnetb0" "efficientnet_b0" "densenet121" "swin" "vit")

num_epochs=(100)


# Iterate over hyperparameter combinations
for model in "${models[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for num_epoch in "${num_epochs[@]}"; do
        run_name="${model}_bs${batch_size}_lr${learning_rate}_epochs${num_epoch}"
        echo "Running: $run_name"

        python main.py \
          --model "$model" \
          --num_epochs "$num_epoch" \
      done
    done
  done
done