 #!/bin/bash --login
 #SBATCH --nodes=1
 #SBATCH --ntasks-per-node=1
 #SBATCH --cpus-per-task=8
 #SBATCH --mem=35G
 #SBATCH --job-name=0.0662_ex_gcn4
 #SBATCH --time=02:00:00
 #SBATCH --partition=gpu_rocm
 #SBATCH --account=a_boden
#SBATCH --gres=gpu:mi210:1 #you can ask for up to 2 here
#SBATCH -o 0.17_ex_gcn4_%A_%a.output
#SBATCH -e 0.17_ex_gcn4_%A_%a.error
#SBATCH --array=1-5

# Module loads #
module load anaconda3/2022.05

#NOTE you will need to make changes here, to match your own conda environments.
source activate /home/s4646506/.conda/envs/seb_rocm

# set working directory 
dir="/scratch/user/s4646506/evoVAE/scripts/"

cd ${dir}

srun python train_vae.py ./configs/gcn4/gcn4_r${SLURM_ARRAY_TASK_ID}_extant_0.0662.yaml
