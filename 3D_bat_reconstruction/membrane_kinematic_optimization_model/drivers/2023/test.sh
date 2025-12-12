#!/bin/bash
#SBATCH --account=bat_flight_kinematics   # Account allocation
#SBATCH --partition=l40s_normal_q
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0   # Request one GPU (only valid on GPU partitions)
module load Miniconda3/24.7.1-0
module load CUDA/12.1.1

source activate softRas

python Brunei_2023_bat_15_1_driver.py