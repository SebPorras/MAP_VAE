#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=MPI-Test
#SBATCH --time=0:02:00
#SBATCH --partition=general
#SBATCH --account=a_boden
#SBATCH -o slurm.output
#SBATCH -e slurm.error

# module loads 
module use /home/s4646506/EasyBuild/modules/all
module load iq-tree/2.2.2.6-foss-2022a

srun iqtree2-mpi -nt 2 -s example.phy -redo
