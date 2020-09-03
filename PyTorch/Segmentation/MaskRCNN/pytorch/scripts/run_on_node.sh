#!/bin/bash

# Default values
NGPUS=4
NNODES=1
MASTER=""
CONFIG="configs/e2e_mask_rcnn_R_50_FPN_1x_1GPU.yaml"
RESULTS='results'
LOGFILE='joblog.log'

while getopts ":hn:N:m:c:r:" opt; do
  case $opt in
    h) echo "-h          Display this help message"
       echo "-n [ngpus]  Number of GPUs to use on each node (default 2)"
       echo "-N [nodes]  Number of nodes to use (default 1)"
       echo "-m [mastr]  Address of master node. Only used if NNODES>1"
       echo "-c [cfg]    Config file (default configs/e2e_mask_rcnn_R_50_FPN_1x.yaml)"
       echo "-r [result] Results dir (default results)"
       exit 0
    ;;
    n) NGPUS="$OPTARG"
    ;;
    N) NNODES="$OPTARG"
    ;; 
    m) MASTER="$OPTARG"
    ;;
    c) CONFIG="$OPTARG"
    ;;
    r) RESULTS="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

#module load xl spectrum_mpi
#module load conda
#source /scratch/apps/conda/4.8.3/bin/activate pytorch1.2
#export LD_PRELOAD=/opt/ibm/spectrum_mpi/lib/pami_451/libpami.so:$LD_PRELOAD
#LOCAL_RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$MV2_COMM_WORLD_RANK

mkdir -p $RESULTS

echo "Using config $CONFIG"
echo NGPUS: $NGPUS NNODES: $NNODES MASTER:$MASTER

if [ $NNODES -eq 1 ]; then
  time python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
    tools/train_net.py \
      --config-file $CONFIG \
      --kfac \
      PER_EPOCH_EVAL True \
      MIN_BBOX_MAP 0.377 \
      MIN_MASK_MAP 0.342 \
      OUTPUT_DIR $RESULTS \
      DTYPE "float32" \
      | tee "${RESULTS}/${LOGFILE}"
else
  echo "Running on rank $LOCAL_RANK"
  time python -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \
      --nnodes=$NNODES \
      --node_rank=$LOCAL_RANK \
      --master_addr="$MASTER" \
    tools/train_net.py \
      --config-file $CONFIG \
      --kfac \
      PER_EPOCH_EVAL True \
      MIN_BBOX_MAP 0.377 \
      MIN_MASK_MAP 0.342 \
      DTYPE "float32" \
      OUTPUT_DIR $RESULTS \
      | tee "${RESULTS}/${LOGFILE}"
fi

