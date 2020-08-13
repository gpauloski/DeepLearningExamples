#!/bin/bash

HOSTFILE=/tmp/hostfile

MASTER=$(head -n 1 $HOSTFILE)
NGPUS=4
NNODES=$(wc -l $HOSTFILE)
CONFIG="configs/e2e_mask_rcnn_R_50_FPN_1x_16GPU.yaml"

mpirun -hostfile $HOSTFILE -N 1 ./run_on_node.sh -n $NGPUS -N $NNODES -m $MASTER -c $CONFIG
