#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J sbatch/maskrcnn                 # Job name
#SBATCH -o sbatch/maskrcnn_kfac.o%j # Name of stdout output file
#SBATCH -N 4                       # Total # of nodes
#SBATCH -n 16                       # Total # of mpi tasks
#SBATCH -t 24:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-user=jgpauloski@utexas.edu
#SBATCH --mail-type=end            # Send email at begin and end of job
#SBATCH -p v100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

#source $SCRATCH/anaconda3/bin/activate pytorch1.2

#scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile4

#cat /tmp/hostfile

#mpiexec -hostfile /tmp/hostfile -N 1  ../cp_coco_to_tmp.sh

mpiexec -hostfile /tmp/hostfile4 -N 4 python \
  tools/train_net.py \
    --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_16GPU.yaml" \
    --kfac \
    PER_EPOCH_EVAL True \
    MIN_BBOX_MAP 0.377 \
    MIN_MASK_MAP 0.342 \
    OUTPUT_DIR "results/kfac_default_16gpu" \
    | tee "joblogs/kfac_default_16gpu.log"
