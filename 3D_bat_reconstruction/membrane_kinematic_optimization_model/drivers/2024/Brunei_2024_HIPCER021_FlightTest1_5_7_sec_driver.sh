#!/bin/bash
#SBATCH -J Brunei_2024_HIPCER021_FlightTest1_5_7_sec_driver_membrane_opt   # Name of the job
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

python HIPCER021_FlightTest1_5_7_sec_driver.py