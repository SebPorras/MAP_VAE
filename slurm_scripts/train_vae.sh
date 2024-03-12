#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --job-name=seb_vae
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=0:30:00
#SBATCH -o vae_train_%j.output
#SBATCH -e vae_train_%j.error
#SBATCH --mail-user=s.porras@uq.edu.au
#SBATCH --mail-type=END
#SBATCH --partition=gpu_cuda
#SBATCH --account=a_boden

# Module loads #
module load anaconda3/2022.05

#NOTE you will need to make changes here, to match your own conda environments.
source activate /home/s4646506/.conda/envs/seb_cuda

# set working directory 
dir="/scratch/user/s4646506/vae_training"
#mkdir -p ${dir}
cd ${dir}

srun python run_vae.py
