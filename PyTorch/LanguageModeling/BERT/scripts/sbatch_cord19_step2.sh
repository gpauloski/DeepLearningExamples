#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J bertkfc                 # Job name
#SBATCH -o sbatch_logs/bertkfc.o%j # Name of stdout output file
#SBATCH -N 16                      # Total # of nodes 
#SBATCH -n 32                      # Total # of mpi tasks
#SBATCH -t 48:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-user=jgpauloski@utexas.edu
#SBATCH --mail-type=end            # Send email at begin and end of job
#SBATCH -p v100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

mkdir -p sbatch_logs

module load conda
module unload spectrum_mpi
module use /home/01255/siliu/mvapich2-gdr/modulefiles/
module load gcc/7.3.0 
module load mvapich2-gdr/2.3.4
conda activate pytorch

export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=1
export MV2_THREADS_PER_PROCESS=2
export MV2_SHOW_CPU_BINDING=1
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_HYBRID_BINDING_POLICY=spread
export MV2_USE_RDMA_CM=0
export MV2_SUPPORT_DL=1

HOSTFILE=hostfile
if [ ! -z "$SLURM_NODELIST" ] ; then
    scontrol show hostnames $SLURM_NODELIST > $HOSTFILE
fi
cat $HOSTFILE

MASTER_RANK=$(head -n 1 $HOSTFILE)
NODES=$(< $HOSTFILE wc -l)
PROC_PER_NODE=4

mpirun_rsh --export-all -np $NODES -hostfile $HOSTFILE \
    bash scripts/run_cord19_step2.sh  --ngpus $PROC_PER_NODE --nnodes $NODES \
        --master $MASTER_RANK --output results/cord19_bert_mini --kfac true --resume true --mvapich
