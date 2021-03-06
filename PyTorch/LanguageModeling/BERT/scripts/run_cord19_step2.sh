#!/bin/bash

NGPUS=1
NNODES=1
MASTER=""
KFAC="false"
OUTPUT=results
MVAPICH=false

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        -h|--help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  -h,--help          Display this help message"
            echo "  -N,--ngpus [int]   Number of GPUs per node (default: 1)"
            echo "  -n,--nnodes [int]  Number of nodes this script is launched on (default: 1)"
            echo "  -m,--master [str]  Address of master node (default: \"\")"
            echo "  --kfac   [bool]    Address of master node (default: false)"
            echo "  --output [path]    Address of master node (default: results)"
            echo "  --mvapich          Use MVAPICH env variables for initialization (default: false)"
            exit 0
        ;;
        -N|--ngpus)
            NGPUS=$VALUE
        ;;
        -n|--nnodes)
            NNODES=$VALUE
        ;;
        -m|--master)
            MASTER=$VALUE
        ;;
        --kfac)
            KFAC=$VALUE
        ;;
        --output)
            OUTPUT=$VALUE
        ;;
        --resume)
            RESUME=$VALUE
        ;;
        --mvapich)
            MVAPICH=true
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done

if [ "$MVAPICH" == true ]; then
  LOCAL_RANK=$MV2_COMM_WORLD_RANK
else
  LOCAL_RANK=$OMPI_COMM_WORLD_RANK
fi

#export NCCL_DEBUG=INFO

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, kfac=$KFAC, output=$OUTPUT, using_mvapich=$MVAPICH

KWARGS=""
if [ "$KFAC" == "true" ] ; then
  KWARGS+=" --kfac"
  KWARGS+=" --kfac_cov_update_freq 1"
  KWARGS+=" --kfac_update_freq 10"
  KWARGS+=" --stat_decay 0.95"
  KWARGS+=" --damping 0.001"
  KWARGS+=" --kl_clip 0.001"
fi

if [ "$RESUME" == "true" ] ; then
  KWARGS+=" --resume_from_checkpoint"
fi

which python

GLOBAL_BATCH_SIZE=16384
GLOBAL_NGPUS=$(($NGPUS * $NNODES))
ACCUMULATED_BATCH_SIZE=$(($GLOBAL_BATCH_SIZE / $GLOBAL_NGPUS))
PER_GPU_BATCH_SIZE=2
ACCUMULATION_STEPS=$(($ACCUMULATED_BATCH_SIZE / $PER_GPU_BATCH_SIZE))

python -m torch.distributed.launch \
   --nproc_per_node=$NGPUS \
   --nnodes=$NNODES \
   --node_rank=$LOCAL_RANK \
   --master_addr=$MASTER \
 run_pretraining.py \
    --seed=5632 \
    --train_batch_size=$ACCUMULATED_BATCH_SIZE \
    --learning_rate=4e-3 \
    --input_dir=data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/cord19 \
    --bert_model=bert-base-uncased \
    --max_seq_length=512 \
    --max_steps=1563 \
    --max_predictions_per_seq=80 \
    --num_steps_per_checkpoint=100 \
    --warmup_proportion=0.128 \
    --do_train \
    --gradient_accumulation_steps=$ACCUMULATION_STEPS \
    --config_file=bert_cord19_config.json \
    --output_dir=$OUTPUT \
    --fp16 \
    --phase2 \
    --phase1_end_step=4000 \
    $KWARGS
