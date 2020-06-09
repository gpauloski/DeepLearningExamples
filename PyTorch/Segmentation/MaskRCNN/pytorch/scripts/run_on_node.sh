#!/bin/bash

# Default values
NGPUS=2
NNODES=1
MASTER=""
BATCH_SIZE=4
MAX_ITER=720000
STEPS="(480000,640000)"
LR=0.0025
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
RESULTS='results'
LOGFILE='joblog.log'

while getopts ":hcn:N:m:b:i:s:l:" opt; do
  case $opt in
    h) echo "-h         Display this help message"
       echo "-n [ngpus] Number of GPUs to use on each node (default 2)"
       echo "-N [nodes] Number of nodes to use (default 1)"
       echo "-m [mastr] Address of master node. Only used if NNODES>1"
       echo "-b [batch] Total batch size (not batch size per gpu or node) (default 4)"
       echo "-i [miter] Max iterations (default 720000)"
       echo "-s [steps] Iteration scheduling ex) \"(400,600\" (480000, 640000)"
       echo "-l [lrate] Learnig rate (default 0.0025)"
       exit 0
    ;;
    n) NGPUS="$OPTARG"
    ;;
    N) NNODES="$OPTARG"
    ;; 
    m) MASTER="$OPTARG"
    ;;
    b) BATCH_SIZE="$OPTARG"
    ;;
    i) MAX_ITER="$OPTARG"
    ;;
    s) STEPS="$OPTARG"
    ;;
    l) LR="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

pushd pytorch

if [ $NNODES -eq 1 ]; then
  time python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
    tools/train_net.py \
      --config-file $CONFIG \
      SOLVER.IMS_PER_BATCH $BATCH_SIZE \
      TEST.IMS_PER_BATCH $(($NGPUS*$NNODES)) \
      SOLVER.MAX_ITER $MAX_ITER \
      SOLVER.STEPS $STEPS \
      SOLVER.BASE_LR $LR \
      OUTPUT_DIR $RESULTS \
      DTYPE "float16" \
      | tee $LOGFILE
else
  echo "Running on rank $PMI_RANK"
  time python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
      --nnodes=$NNODES \
      --node_rank=$PMI_RANK \
      --master_addr="$MASTER" \
    tools/train_net.py \
      --config-file $CONFIG \
      SOLVER.IMS_PER_BATCH $BATCH_SIZE \
      TEST.IMS_PER_BATCH $(($NGPUS*$NNODES)) \
      SOLVER.MAX_ITER $MAX_ITER \
      SOLVER.STEPS $STEPS \
      SOLVER.BASE_LR $LR \
      DTYPE "float16" \
      OUTPUT_DIR $RESULTS \
      | tee $LOGFILE
fi

popd

