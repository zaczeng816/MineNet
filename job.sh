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
  --overlay /scratch/wz1492/overlay-25GB-500K.ext3:rw \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /scratch/wz1492/env.sh;"

source /scratch/wz1492/miniconda3/etc/profile.d/conda.sh
conda activate Vim

models=( "resnet50" "densenet121" "efficientnet_b0" "vit" "swin" "mamba" )



# Iterate over hyperparameter combinations
for model in "${models[@]}"; do
  run_name="${model}_epochs${num_epoch}"
  echo "Running: $run_name"

  python main.py --model "$model" --epochs 80 --bands "0,1,2,3,4,5,6,7,8,9,10,11" 
  python main.py --model "$model" --epochs 80 --bands "0,1,2" 
done

