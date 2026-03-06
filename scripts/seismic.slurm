#!/bin/bash
#SBATCH --job-name=seism_train
#SBATCH --output=logs/csgm_train_%j.out
#SBATCH --error=logs/csgm_train_%j.err
#SBATCH --time=24:00:00           
#SBATCH --partition=highgpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=davide.sabeddu@ucf.edu
#SBATCH --export=NONE

# Create logs directory if it doesn't exist
mkdir -p logs

# Check CUDA availability
~/.conda/envs/csgm/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run the training script
~/.conda/envs/csgm/bin/python /home/da389032/csgm/scripts/train_conditional_seismic_imaging.py \
--experiment_name "seismic_imaging_fullset" \
--phase "train" \
--max_epochs 3000 \
--save_freq 100 \
