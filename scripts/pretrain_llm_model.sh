#!/bin/sh

accelerate launch \
  --config_file configs/accelerate_configs/ds_stage1.yaml \
  TrainModel/pretrain_llm_model.py \
  --train_config configs/pretrain_configs/chatglm.yaml \
  --model_config model/llm/chatglm2
