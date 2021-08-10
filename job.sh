#!/bin/bash
#SBATCH -J mnieto_test_eval
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g

#SBATCH -o /homedtic/mnieto/test_eval/%N.%J.mnieto_test_loader.out
#SBATCH -e /homedtic/mnieto/test_eval/%N.%J.mnieto_test_loader.err

module load CUDA/11.0.3

export PATH="$HOME/project/anaconda3/bin:$PATH"
export PATH="$/homedtic/mnieto/project/anaconda3/envs/torch_thesis:$PATH"
source activate torch_thesis
cd /homedtic/mnieto/project/TransformerGrooveTap2Drum/model/
export WANDB_API_KEY=""
python -m wandb login
wandb agent marinaniet0/test_sweep_t2d/qyp15cdc
#python train_tap2drum.py
