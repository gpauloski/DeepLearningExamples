#!/bin/bash

MASTER=$(head -n 1 /tmp/hostfile)
NGPUS=4
NNODES=2
BATCH_SIZE=16
MAX_ITER=45000
STEPS="(30000,40000)"
LR=0.04

mpirun -hostfile /tmp/hostfile -N $NGPUS \
  ./run_on_node.sh -n $NGPUS -N $NNODES -m $MASTER -b $BATCH_SIZE \
           -i $MAX_ITER -s $STEPS -l $LR
