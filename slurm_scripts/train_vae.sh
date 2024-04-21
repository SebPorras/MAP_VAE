#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --job-name=seb_train_vae
#SBATCH --time=1:30:00
#SBATCH --partition=gpu_rocm
#SBATCH --account=a_boden
#SBATCH --gres=gpu:mi210:1 #you can ask for up to 2 here
#SBATCH -o gfp_vae.output
#SBATCH -e gfp_vae.error
  
# Module loads #
module load anaconda3/2022.05

#NOTE you will need to make changes here, to match your own conda environments.
source activate /home/s4646506/.conda/envs/seb_rocm

# set working directory 
dir="/scratch/user/s4646506/evoVAE/scripts/"
cd ${dir}

srun python train_seqVAE.py