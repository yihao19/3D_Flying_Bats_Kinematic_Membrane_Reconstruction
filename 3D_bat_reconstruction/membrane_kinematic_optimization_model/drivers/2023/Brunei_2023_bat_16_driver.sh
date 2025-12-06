#!/bin/bash
#SBATCH -J Brunei_2023_bat_16_driver_membrane   # Name of the job
#SBATCH --account=bat_flight_kinematics   # Account allocation
#SBATCH --partition=v100_normal_q   # Partition of the cluster
#SBATCH --nodes=1   # Number of compute nodes
#SBATCH --ntasks-per-node=1   # Number of processes
#SBATCH --cpus-per-task=1   # Number of CPU cores per process
#SBATCH --time=6-00:00:00   # Runtime limit of 10 minutes
#SBATCH --gres=gpu:1   # Request one GPU (only valid on GPU partitions)


module load Miniconda3/24.7.1-0
module load CUDA/12.1.1

source activate softRas

python /home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/drivers/2023/Brunei_2023_bat_16_driver.py