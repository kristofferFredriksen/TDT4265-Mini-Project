#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --job_name=tdt4265_yolo_first_run
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slim-%j.err
#SBATCH --mail-user=kristotf@stud.ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "JOB_START $(date -Is)"
echo "JOBID $SLURM_JOB_ID"
echo "NODELIST $SLURM_JOB_NODELIST"
echo "PWD $(pwd)"

module purge
module load Anaconda3/2024.02-1

source ~/bashrc || true
conda activate tdt4265

mkdir -p runs/cluster_logs

GPU_LOG="runs/cluster_logs/gpu_power_${SLURM_JOB_ID}.txt"
nvidia-smi. --query-gpu=timestamp,name,power.draw,utilization.gpu,temperature.gpu \
--format=csv,noheader,nounits -l 1 > "${GPU_LOG} & SMI_PID=$!

TIME_LOG="rund/cluster_logs/time_${SLURM_JOB_ID}.txt
/usr/bin/time -v python train.py 2> "${TIME_LOG}"

kill "${SMI_PID} || true

echo "JOB_END $(date -Is)

sacct -j "${SLURM_JOB_ID}" \
  --format=JobID,JobName%20,Partition,Elapsed,TotalCPU,MaxRSS,AllocTRES%40,ConsumedEnergy,ConsumedEnergyRaw \
  > "runs/cluster_logs/sacct_${SLURM_JOB_ID}.txt" || true