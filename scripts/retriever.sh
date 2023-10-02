#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=lizx2333  # change to your wandb account
export WANDB_API_KEY=209d2c11abb1119afd7b6e1fc4271b1d52f2918c  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="4,5,6,7"

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
  index_data=index_data/${task_name}/svamp/train.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  retrieve_file=${run_dir}/retrieved.json




  scored_file=${run_dir}/scored.json



  

  run_name=bert-fix_ctx-shared-bs64/lr1e-6epoch_qa_model
  run_dir=${run_dir}/${run_name}
  accelerate launch  --main_process_port ${port}  qa_retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${scored_file} \
      qa_dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      model_config.ctx_model_name=null  # share ctx model with q model


done


