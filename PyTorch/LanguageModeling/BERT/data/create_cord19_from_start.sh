#!/bin/bash

export BERT_PREP_WORKING_DIR='.'
DATASET=cord19
VOCAB_FILE=../vocab/cord19_vocab.txt

python bertPrep.py --action download --dataset $DATASET
python bertPrep.py --action text_formatting --dataset $DATASET
python bertPrep.py --action sharding --dataset $DATASET

# Create HDF5 files Phase 1
python bertPrep.py --action create_hdf5_files --dataset $DATASET \
	--max_seq_length 128 --max_predictions_per_seq 20 \
	--vocab_file $VOCAB_FILE --do_lower_case 1

# Create HDF5 files Phase 2
python bertPrep.py --action create_hdf5_files --dataset $DATASET \
	--max_seq_length 512 --max_predictions_per_seq 80 \
	--vocab_file $VOCAB_FILE --do_lower_case 1
