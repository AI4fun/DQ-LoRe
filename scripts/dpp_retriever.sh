#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=Your WANDB name  # change to your wandb account
export WANDB_API_KEY=Your API key  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"
#export CUDA_VISIBLE_DEVICES="4,5,6,7"

gpu=4
method=epr
num_ice=8
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
  #index_data=index_data/${task_name}/svamp/train.json
  index_data=index_data/${task_name}/length_index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  retrieve_file=${run_dir}/retrieved.json




  scored_file=${run_dir}/scored.json



  run_name=bert-fix_ctx-shared-bs64
  run_dir=${run_dir}/${run_name}


  retrieve_file=${run_dir}/train_retrieved_ceil.json
  python dense_retriever.py \
      output_file=${retrieve_file} \
      hydra.run.dir=${run_dir}/dense_retriever \
      num_ice=${num_ice} \
      task_name=${task_name} \
      model_config.norm_embed=true \
      index_reader.dataset_path=${index_data} \
      dataset_reader.dataset_split=test \
      faiss_index=${run_dir}/ceil_index \
      dpp_search=true \
      model_config.scale_factor=0.1 \
      dpp_topk=100 \
      mode=map


done


