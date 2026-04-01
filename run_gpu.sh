#!/bin/bash
#SBATCH --job-name=nim
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=benv-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export TOKENIZERS_PARALLELISM=false


module reset
module load python/miniforge3_pytorch/2.7.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nim-env
cd /u/iyu1/nim_game_project/access_files

timeout 48h python -u dann_meanpool.py $1
