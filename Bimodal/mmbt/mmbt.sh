#!/bin/bash
python mmbt/train.py --batch_sz 4 --gradient_accumulation_steps 40 \
 --savedir /path/to/savedir/ --name mmbt_model_run \
 --data_path /path/to/datasets/ \
 --task mmimdb --task_type classification \
 --model mmbt --num_image_embeds 1 --freeze_txt 0.1 --freeze_img 0.3  \
 --patience 10 --dropout 0.2 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1
