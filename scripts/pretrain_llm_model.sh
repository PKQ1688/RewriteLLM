#!/bin/sh

accelerate launch \
  --config_file configs/accelerate_configs/ds_stage1.yaml \
  --main_process_port 27643 TrainModel/pretrain_llm_model.py \
  --train_config /sft_configs/llama.yaml \
  --model_config openlm-research/open_llama_7b_v2
