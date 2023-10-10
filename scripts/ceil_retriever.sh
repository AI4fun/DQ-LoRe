#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=Your WANDB name  # change to your wandb account
export WANDB_API_KEY=Your API key  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES="0"

gpu=1
method=epr
num_ice=50
port=5324

#model_name=gpt2-large
#n_tokens=700
#scr_batch_size=128
#inf_batch_size=48

model_name=EleutherAI/gpt-neo-2.7B
n_tokens=800
scr_batch_size=2
inf_batch_size=2

for task_name in svamp
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/svamp/train.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}
  epr_model=output/epr/${task_name}/${model_name}/bert-fix_ctx-shared-bs64

  retrieve_file=${run_dir}/retrieved.json




  scored_file=${run_dir}/scored.json


  scale_factor=0.1
  run_name=base-mg0.02-s${scale_factor}-fix
  run_dir=${run_dir}/${run_name}
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      pair_wise=true \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      pretrained_model_path=${epr_model} \
      training_args.num_train_epochs=30 \
      training_args.per_device_train_batch_size=64 \
      training_args.per_device_eval_batch_size=64 \
      training_args.gradient_accumulation_steps=1 \
      model_config.dpp_training=true \
      model_config.norm_embed=true \
      model_config.margin=0.02 \
      model_config.scale_factor=${scale_factor}


done


