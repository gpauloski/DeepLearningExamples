#!/bin/bash

scontrol show hostnames $SLURM_NODELIST > /tmp/hostfile
cat /tmp/hostfile

MASTER=$(head -n 1 /tmp/hostfile)
NGPUS=4
NNODES=$(< /tmp/hostfile wc -l)
CONFIG="configs/e2e_mask_rcnn_R_50_FPN_1x_8GPU_32.yaml"

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

mpirun_rsh --export-all -np $NNODES -hostfile /tmp/hostfile \
    ./scripts/run_on_node.sh -n $NGPUS -N $NNODES -m $MASTER -c $CONFIG -r results_kfac_batch_norm
