#!/bin/bash
#SBATCH --job-name=1-gpu
#SBATCH --output=logs/finetune-gpu-%j.out
#SBATCH --error=logs/finetune-gpu-%j.err
#SBATCH --partition=gpuA40x4          # use the GPU partition
#SBATCH --gres=gpu:1                  # request 1 GPU
#SBATCH --account=benv-delta-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

module load anaconda3_gpu/23.9.0
#module load cuda                      # ensure CUDA drivers/toolkit are loaded
conda activate hf-nim        # e.g. 'nim-gpu'
pip install torch datasets huggingface_hub
cd /u/lvillani/nim_game_project

#echo "=== Starting NVIDIA-SMI monitoring in background ==="
#nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 60 > logs/gpu_usage_$SLURM_JOB_ID.log &
#NVIDIA_MON_PID=$!


timeout 48h python finetune_nim.py
#timeout 5h python test_model.py
#timeout 30h python test_multi.py
#timeout 12h python test_model_maxrem.py
#timeout 30h python double_finetune.py


#echo "=== Stopping NVIDIA-SMI monitoring ==="
#kill $NVIDIA_MON_PID
