#!/bin/bash
#SBATCH --job-name=kek     # create a short name for your job
#SBATCH --partition=rnd
#SBATCH --exclude=99dgx-09,99dgx-08
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
##SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_outputs/%j.out  # to write outputs
#SBATCH -e slurm_outputs/%j.err  # to write errs


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


# to add conda stuff
# source /home/${USER}/.bashrc
source /home/i.sedyh/miniconda3/bin/activate asr_env

unset TMPDIR


# to use ALL CHECKPOINTS
#CHECKPOINT_DIR=outputs/2023-12-02/22-08-10/output/checkpoints

# to use specific one
CHECKPOINT_DIR=outputs/2023-12-15/18-16-43/output/checkpoints/step-30000
# or 
#CHECKPOINT_DIR=checkpoints/step-190000/model.safetensors

srun \
python synthesise.py \
    +test_audio_dir=test_data \
    +checkpoint_dir=$CHECKPOINT_DIR \
# hydra.job.chdir=False