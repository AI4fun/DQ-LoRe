#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=Your WANDB name  # change to your wandb account
export WANDB_API_KEY=Your API key  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

gpu=4
method=DQ-LoRe
num_ice=50
port=5324

#model_name=gpt2-large
#n_tokens=700
#scr_batch_size=128
#inf_batch_size=48

model_name=text-davinci-003
n_tokens=800
scr_batch_size=2
inf_batch_size=2

for task_name in gsm8k
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  retrieve_file=${run_dir}/retrieved.json
  python bm25_retriever.py \
      hydra.run.dir=${run_dir}/bm25_retriever \
      output_file=${retrieve_file} \
      num_candidates=50 \
      num_ice=1 \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      dataset_split=train \
      ds_size=44000 \
      query_field=a \
      index_reader.field=a



  scored_file=${run_dir}/scored.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  api_scorer.py \
      hydra.run.dir=${run_dir}/scorer \
      task_name=${task_name} \
      output_file=${scored_file} \
      batch_size=${scr_batch_size} \
      model_name=${model_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data}


  run_name=bert-fix_ctx-shared-bs64
  run_dir=${run_dir}/${run_name}
  pretrained_model=${run_dir}/qa_model
  accelerate launch  --main_process_port ${port}  qa_retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      model_config.ctx_model_name=null  # share ctx model with q model
      pretrained_model=${pretrained_model}


  retrieve_file=${run_dir}/qa_train_retrieved.json
  pred_file=${run_dir}/pred.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  qa_inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      dataset_reader.dataset_split=test \
      batch_size=${inf_batch_size} \
      retrieve_file=${retrieve_file} \
      pretrained_model=${pretrained_model} \
      pca_num=128
done
