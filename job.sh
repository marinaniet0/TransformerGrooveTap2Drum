#!/bin/bash
  #SBATCH -J mnieto_test_loader # your name for the job
  #SBATCH -p short # medium high or low
  #SBATCH -N 1 # number of nodes
  #SBATCH --workdir=/homedtic/mnieto/ # Working directory
  #SBATCH --gres=gpu:1 # GPU request
  #SBATCH --mem=8g # CPU request
  #SBATCH -o /homedtic/mnieto/%N.%J.mnieto_test_loader.out # STDOUT # extract output from the running file to this directory
  #SBATCH -e /homedtic/mnieto/%N.%J.mnieto_test_loader.err # STDOUT # extract errors from the running file to this directory

  module load CUDA/11.0.3

  export PATH="$HOME/project/anaconda3/bin:$PATH"
  export PATH="$/homedtic/mnieto/project/anaconda3/envs/torch_thesis:$PATH"
  source activate torch_thesis
  cd /homedtic/mnieto/project/TransformerGrooveTap2Drum/model/
  python experiment.py