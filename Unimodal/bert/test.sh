#!/bin/sh

export BERT_BASE_DIR=~/vishal/bert_hum/bert
export TRAINED_CLASSIFIER=./bert_output/model.ckpt-524
python3 ~/vishal/bert_hum/bert/run_classifier.py \
--task_name=cola \
--do_predict=true \
--data_dir=./data1\
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$TRAINED_CLASSIFIER \
--max_seq_length=64 \
--output_dir=./bert_output/
