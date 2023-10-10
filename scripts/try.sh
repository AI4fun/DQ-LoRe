#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=lizx2333  # change to your wandb account
export WANDB_API_KEY=209d2c11abb1119afd7b6e1fc4271b1d52f2918c  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

#export CUDA_VISIBLE_DEVICES="1,0,2,3"
export CUDA_VISIBLE_DEVICES="2,5,6,7"

gpu=4
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
  #index_data=index_data/${task_name}/length_index_dataset.json
  index_data=index_data/${task_name}/svamp/train.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  retrieve_file=${run_dir}/retrieved.json




  scored_file=${run_dir}/scored.json



  run_name=bert-fix_ctx-shared-bs64
  run_dir=${run_dir}/${run_name}



  retrieve_file=${run_dir}/train_retrieved.json


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
      batch_size=${inf_batch_size}
      retrieve_file=${retrieve_file}
done


