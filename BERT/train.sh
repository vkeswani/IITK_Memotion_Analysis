#!/bin/sh

export BERT_BASE_DIR=~/vishal/bert_hum/bert
python3 ~/vishal/bert_hum/bert/run_classifier.py \
--task_name=cola \
--do_train=true \
--do_eval=true \
--data_dir=./data1 \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=64 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--output_dir=./bert_output/
