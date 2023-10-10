import glob
import json
import os
import logging
import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed
from src.metrics import get_metric
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.utils.statistics import show_statistics
from src.models.api_client import run_api
from src.utils.misc import parallel_run, save_json
from src.models.model import ppl_generate


import json
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
import os
from transformers import set_seed
from torch.utils.data import DataLoader
from src.utils.dpp_map import fast_map_dpp, k_dpp_sampling
from src.utils.misc import parallel_run, partial
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.models.biencoder import BiEncoder

from transformers import BertTokenizer
from transformers import AutoTokenizer

import json
import re
import itertools

logger = logging.getLogger(__name__)


pca_num = 128

class Inferencer:
    def __init__(self, cfg, accelerator=None) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.gen_field = cfg.dataset_reader.field
        
        self.cfg = cfg

        self.accelerator = accelerator
        self.output_file = cfg.output_file
        # OmegaConf DictConfig to dict
        self.generation_kwargs = OmegaConf.to_object(cfg.model_config.generation_kwargs)
        self.evaluator = get_metric(cfg.task_name)
        

        self.qa_dataset_reader = hu.instantiate(cfg.qa_dataset_reader)
        qa_co = DataCollatorWithPaddingAndCuda(tokenizer=self.qa_dataset_reader.tokenizer, device=self.cuda_device)
        self.qa_dataloader = DataLoader(self.qa_dataset_reader, batch_size=cfg.batch_size, collate_fn=qa_co)
        self.num_candidates = 1
        self.num_ice = 8
        self.is_train = cfg.dataset_reader.dataset_split == "train"
        
        
        qa_model_config = hu.instantiate(cfg.qa_model_config)
        self.pretrained_qa_model = cfg.pretrained_model
        
        self.qa_model = BiEncoder.from_pretrained(self.pretrained_qa_model, config=qa_model_config)
        self.qa_model = self.qa_model.to(self.cuda_device)
        self.qa_model.eval()
        
        self.qa_index = self.create_index(cfg)
        
        self.index_reader = hu.instantiate(cfg.index_reader)

        qa_index_reader = hu.instantiate(cfg.qa_index_reader)
        self.qa_index_reader = qa_index_reader
    
        self.model, self.dataloader = self.init_model_dataloader(cfg)

        self.pca_num = cfg.pca_num
        
        

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)

        if self.accelerator.is_main_process:
            logger.info(f"Statistics after sharding: ")
            show_statistics(self.dataset_reader.encoded_dataset, "main dataset")
            show_statistics(self.dataset_reader.index_reader.encoded_dataset, "index dataset")

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model = hu.instantiate(cfg.model_config.model).eval()
        model = self.accelerator.prepare(model)

        if hasattr(model, "module"):
            model = model.module

        return model, dataloader

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        avg_ice_num = 0
        res = []
        for i, entry in enumerate(dataloader):
            #if i >= 50:
            #    break
            metadata = entry.pop("metadata")
            if 'choices' in self.dataset_reader.dataset_wrapper.field_getter:
                # for classification tasks, we compare the ppl of provided generation_choices as generation
                choices = [self.dataset_reader.dataset_wrapper.get_field(meta, 'choices') for meta in metadata]
                choices_list = list(zip(*choices))
                preds = ppl_generate([meta['prompt'] for meta in metadata],
                                     model=self.model,
                                     tokenizer=self.dataset_reader.tokenizer,
                                     choices_list=choices_list,
                                     device=self.accelerator.device)
                for mdata, pred in zip(metadata, preds):
                    mdata['generated'] = pred
                    avg_ice_num += len(mdata['ice_prompts_list'])
            else:
                with torch.no_grad():
                    outputs = self.model.generate(input_ids=entry.input_ids,
                                                  attention_mask=entry.attention_mask,
                                                  eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                                  pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                                  do_sample=False,  # always use greedy decode here
                                                  **self.generation_kwargs)
                    prompt_len = int(entry.attention_mask.shape[1])
                    for mdata, output in zip(metadata, outputs.tolist()):
                        generated = self.dataset_reader.tokenizer.decode(output[prompt_len:])
                        mdata['generated'] = generated.strip(self.dataset_reader.tokenizer.pad_token).strip()
                        avg_ice_num += len(mdata['ice_prompts_list'])

            res.extend(metadata)

            if i == 0:
                logger.info(f"Prompt: {metadata[0]['prompt']}")
                logger.info(f"Generated: {metadata[0]['generated']}")
                logger.info(f"Number of ICE: {len(metadata[0]['ice_prompts_list'])}")

        save_json(f"{self.output_file}tmp_{self.accelerator.device}.bin", res)

        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num / len(res)}")

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with open(path) as f:
                data.extend(json.load(f))
        # from src.utils.misc import load_json
        # data = load_json(self.output_file)
        preds = [i['generated'] for i in data]
        metric = self.evaluator.evaluate(preds, data)
        logger.info(f"metric: {str(metric)}")

        save_json(self.output_file, data)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data


class APInferencer(Inferencer):
    
    def init_model_dataloader(self, cfg):
        model = hu.instantiate(cfg.model_config.model)
        dataloader = self.dataset_reader
        return model, dataloader
    
    def create_index(self, cfg):
        logger.info("Building faiss index...")
        qa_index_reader = hu.instantiate(cfg.qa_index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=qa_index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(qa_index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.qa_model.encode(**entry, encode_ctx=True)
            res = res.cpu().detach().numpy()
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        
        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        faiss.write_index(index, 'output/epr/gsm8k/EleutherAI/gpt-neo-2.7B/bert-fix_ctx-shared-bs64/qa_index')
        
        
        #logger.info(f"Saving qa faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        return index

    def forward(self):
        responses_file = "Your responses file"
        with open(responses_file, 'r', encoding='latin-1') as file:         
            content = file.read()
        responses = eval(content)
        
        
        print(f"{responses_file} read responses!")
        
        
        
        res_iterator = iter(responses) 
                            
        qa_res_list = []
        qa_tokenizer = BertTokenizer.from_pretrained(self.pretrained_qa_model)
        for i, (entry, response0, response1) in enumerate(zip(self.qa_dataloader, res_iterator, res_iterator)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                ans0 = response0[0]['text']
                ans1 = response1[0]['text']

                ans0 = re.sub(r"\nA: Let's think step by step.\n", "", ans0)
                ans1 = re.sub(r"\nA: Let's think step by step.\n", "", ans1)
                
                metadata[0]['answer'] = ans0
                metadata[1]['answer'] = ans1
                qa_text_0 = metadata[0]['text'] + ans0
                qa_text_1 = metadata[1]['text'] + ans1
                qa_tokenized_inputs_0 = qa_tokenizer.encode_plus(qa_text_0, truncation=True, add_special_tokens=False, return_tensors='pt', padding=True, max_length=512)
                qa_tokenized_inputs_1 = qa_tokenizer.encode_plus(qa_text_1, truncation=True, add_special_tokens=False, return_tensors='pt', padding=True, max_length=512)
                
                input_ids1 = qa_tokenized_inputs_0['input_ids'][0]
                input_ids2 = qa_tokenized_inputs_1['input_ids'][0]
                
                input_att1 = qa_tokenized_inputs_0['attention_mask'][0]
                input_att2 = qa_tokenized_inputs_1['attention_mask'][0]
                
                ids_padded_length = max(len(input_ids1), len(input_ids2))
                padded_input_ids1 = torch.nn.functional.pad(input_ids1, pad=(0, ids_padded_length - len(input_ids1)), value=0)
                padded_input_ids2 = torch.nn.functional.pad(input_ids2, pad=(0, ids_padded_length - len(input_ids2)), value=0)
                
                att_padded_length = max(len(input_att1), len(input_att2))
                padded_input_att1 = torch.nn.functional.pad(input_att1, pad=(0, att_padded_length - len(input_att1)), value=0)
                padded_input_att2 = torch.nn.functional.pad(input_att2, pad=(0, att_padded_length - len(input_att2)), value=0)
                
                qa_inputs_ids = torch.cat((padded_input_ids1.unsqueeze(0), padded_input_ids2.unsqueeze(0)), dim=0).to(self.cuda_device)
                qa_attention_mask = torch.cat((padded_input_att1.unsqueeze(0), padded_input_att2.unsqueeze(0)), dim=0).to(self.cuda_device)
                qa_entry = {'input_ids': qa_inputs_ids, 'attention_mask': qa_attention_mask}
                qa_res = self.qa_model.encode(**qa_entry, **{})
            qa_res = qa_res.cpu().detach().numpy()
            qa_res_list.extend([{"embed": r, "metadata": m} for r, m in zip(qa_res, metadata)])
            
            
        for res in qa_res_list:
            res['entry'] = self.qa_dataset_reader.dataset_wrapper[res['metadata']['id']]
        
        func = partial(pca, num_candidates=self.num_candidates, num_ice=self.num_ice, index_reader = self.qa_index_reader, model = self.qa_model, device = self.cuda_device, pca_num = self.pca_num)
        qa_data = parallel_run(func=func, args_list=qa_res_list, initializer=set_global_object,
                            initargs=(self.qa_index, self.is_train))
        with open(self.cfg.retrieved_path, "w") as f:
            json.dump(qa_data, f)
        
         
        retrieved_dataloader = hu.instantiate(self.cfg.retrieved_reader)
            
        qa_prompts = [entry['metadata']['prompt'] for entry in retrieved_dataloader]
        print(f"================= qa_prompts[0]:{qa_prompts[0]}")
        #assert 1==0
        responses = parallel_run(run_api, args_list=qa_prompts,
                                 n_processes=16,
                                 client=self.model,
                                 **self.generation_kwargs)  
                                 
        print("==================QA APInferencer responses:{}".format(responses))  

        data = []
        
        for i, (entry, response) in enumerate(zip(retrieved_dataloader, responses)):
            entry['metadata']['generated'] = response[0]['text']
            data.append(entry['metadata'])

        save_json(self.output_file, data)

        avg_ice_num = sum([len(i['ice_prompts_list']) for i in data])/len(data)
        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num}")
        preds = [i['generated'] for i in data]
        metric = self.evaluator.evaluate(preds, data)
        logger.info(f"metric: {str(metric)}")
        






def PCA_svd(X, k, center=True, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.to(device)
    #print("##### X.device:", X.device)
    #print("##### H.device:", H.device)
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    return components

def pca(entry, index_reader, model, device, num_candidates=1, num_ice=1, pca_num=128):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, 16)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    entry_embed = torch.tensor(entry['embed']).to(device)
    pca_index = faiss.IndexIDMap(faiss.IndexFlatIP(pca_num))

    pca_embed_list = [entry_embed]

    for ids in range(0, len(near_ids), 2):
        input_id1 = extend_array_with_zeros(index_reader[near_ids[ids]]['input_ids'], 512)
        attention_mask1 = extend_array_with_zeros(index_reader[near_ids[ids]]['attention_mask'], 512)
        input_id2 = extend_array_with_zeros(index_reader[near_ids[ids + 1]]['input_ids'], 512)
        attention_mask2 = extend_array_with_zeros(index_reader[near_ids[ids + 1]]['attention_mask'], 512)

        input_ids = torch.tensor([input_id1, input_id2]).to(device)
        attention_mask = torch.tensor([attention_mask1, attention_mask2]).to(device)

        new_embed = model.encode(**{'input_ids':input_ids, 'attention_mask':attention_mask}, encode_ctx=True)
        new_embed = new_embed.to(device)

        for n in new_embed:
            pca_embed_list.append(n)

    pca_embed_list = torch.stack(pca_embed_list)
    pca_embed_list = PCA_svd(pca_embed_list, pca_num, device = device).to(torch.float32)
    pca_embed_list = pca_embed_list.cpu().detach().numpy()
    id_list = np.array(near_ids)
    embed_list = np.stack([emb for emb in pca_embed_list[1:]])
    pca_index.add_with_ids(embed_list, id_list)
    pca_embed = np.expand_dims(pca_embed_list[0], axis=0)
    pca_ids = pca_index.search(pca_embed, num_ice)[1][0].tolist()
    entry = entry['entry']
    entry['ctxs'] = pca_ids
    entry['ctxs_candidates'] = [[i] for i in pca_ids[:num_candidates]]
    return entry

def extend_array_with_zeros(arr, target_length):
    if len(arr) >= target_length:
        return arr[:target_length]

    extended_arr = arr + [0] * (target_length - len(arr))
    return extended_arr

""" def modified_gram_schmidt(A):
    A = np.array(A)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].conjugate(), A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R """

def modified_gram_schmidt(vectors):
    vectors = np.array(vectors, dtype=np.float64)  # 将整型列表转换为NumPy数组，并指定数据类型为float64
    num_vectors, vector_length = vectors.shape
    orthogonal_vectors = np.empty_like(vectors)

    norms = []
    for i in range(num_vectors):
        v = vectors[i]

        for j in range(i):
            ortho_vector = orthogonal_vectors[j]
            projection = np.dot(v, ortho_vector) / np.dot(ortho_vector, ortho_vector)
            v -= projection * ortho_vector

        norm = np.linalg.norm(v)
        norms.append(norm)
        if norm > 0:
            orthogonal_vectors[i] = v / norm

    return orthogonal_vectors, norms

def mgs(entry, index_reader, model, device, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, 10)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    mgs_score = []
    combinations = list(itertools.combinations(near_ids, num_ice))
    for comb in combinations:
        #print("comb:", comb)
        #ctx = [extend_array_with_zeros(index_reader[i]['input_ids'], 512) for i in comb]
        #print("################# ctx: ", ctx)
        #print("################# index_reader[0]", index_reader[0])
        #print("################# {'input_ids':torch.tensor(index_reader[i]['input_ids']), 'attention_mask':torch.tensor(index_reader[i]['attention_mask'])}", {'input_ids':torch.tensor(index_reader[0]['input_ids']), 'attention_mask':torch.tensor(index_reader[0]['attention_mask'])})
        embeds = []
        for i in range(0, len(comb), 2):
            input_id1 = extend_array_with_zeros(index_reader[comb[i]]['input_ids'], 512)
            attention_mask1 = extend_array_with_zeros(index_reader[comb[i]]['attention_mask'], 512)
            input_id2 = extend_array_with_zeros(index_reader[comb[i + 1]]['input_ids'], 512)
            attention_mask2 = extend_array_with_zeros(index_reader[comb[i + 1]]['attention_mask'], 512)

            input_ids = torch.tensor([input_id1, input_id2]).to(device)
            attention_mask = torch.tensor([attention_mask1, attention_mask2]).to(device)
            #print("{'input_ids':input_ids, 'attention_mask':attention_mask}:", {'input_ids':input_ids, 'attention_mask':attention_mask})
            
            new_embed = model.encode(**{'input_ids':input_ids, 'attention_mask':attention_mask}, encode_ctx=True)
            new_list = new_embed.tolist()
            for l in new_list:
                embeds.append(l)
        #print("################# embeds", embeds)
        #assert 1==0

        q, norms = modified_gram_schmidt(embeds)
        #print(f"################# q:{q}")
        #print(f"q.shape:{q.shape}")
        #print(f"################# norms:{norms}")
        score = 1
        """ for n in norms:
            score = score * n """
        
        score = sorted(norms, reverse=True)[0]

        mgs_score.append(score)
        #for 
        #assert 1==0
    
    sorted_ids = sorted(enumerate(mgs_score), key=lambda x: x[1], reverse=True)
    sorted_ids = [index for index, _ in sorted_ids]
    #print("sorted_ids:", sorted_ids)
    #print("combinations[sorted_ids[0]]:", combinations[sorted_ids[0]])
    ids = list(combinations[sorted_ids[0]])
    #assert 1==0
    entry = entry['entry']
    entry['ctxs'] = ids

    entry['ctxs_candidates'] = [[i] for i in ids[:num_candidates]]

    return entry
        
def set_global_object(index, is_train):
    global index_global, is_train_global
    index_global = index
    is_train_global = is_train 
        
def far_knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    #near_ids = index_global.search(embed, max(num_candidates, num_ice)+1)[1][0].tolist()
    #near_ids = near_ids[1:] if is_train_global else near_ids
    
    near_ids = index_global.search(embed, 7473)[1][0].tolist()
    far_ids = list(reversed(near_ids))
    
    

    entry = entry['entry']
    #entry['ctxs'] = near_ids[:num_ice]
    entry['ctxs'] = far_ids[:num_ice]
    #entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]
    entry['ctxs_candidates'] = [[i] for i in far_ids[:num_candidates]]
    return entry
    
def knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, max(num_candidates, num_ice)+1)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    entry = entry['entry']
    entry['ctxs'] = near_ids[:num_ice]

    entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]

    return entry
    
def lenth_knn(entry, num_candidates=1, num_ice=1):
    #print("######################### entry:{}".format(entry))
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, 16)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids
    
    answer_len = len(entry['metadata']['answer'].split("\n"))
    #print("######################### entry['metadata']['answer']:{}".format(entry['metadata']['answer']))
    #print("######################### answer_len:{}".format(answer_len))
    
    with open('./index_data/gsm8k/length_index_dataset.json', 'r', encoding='latin-1') as file:
        data = json.load(file)
    
    #print("######################### data[0]['answer']:{}".format(data[0]['answer']))
    lenth = [len(data[i]['answer'].split("\n")) for i in near_ids]    
    len_score = [abs(l - answer_len - 0.3) for l in lenth]
    print("######################### answer_len:{}, lenth:{}, len_score:{}".format(answer_len, lenth, len_score))
    #print("######################### len_score:{}".format(len_score))
    sorted_ids = [x for _, x in sorted(zip(len_score, near_ids))]
    #print("######################### sorted_ids:{}".format(sorted_ids))
    
    assert 1==0
    entry = entry['entry']
    entry['ctxs'] = sorted_ids[:num_ice]

    entry['ctxs_candidates'] = [[i] for i in sorted_ids[:num_candidates]]

    return entry

def sorted_knn(entry, num_candidates=1, num_ice=1):
    #print("######################### entry:{}".format(entry))
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, 16)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids
    
    answer_len = len(entry['metadata']['answer'].split("\n"))
    #print("######################### entry['metadata']['answer']:{}".format(entry['metadata']['answer']))
    #print("######################### answer_len:{}".format(answer_len))
    
    with open('./index_data/gsm8k/index_dataset.json', 'r', encoding='latin-1') as file:
        data = json.load(file)
    
    #print("######################### data[0]['answer']:{}".format(data[0]['answer']))
    lenth = [len(data[i]['answer'].split("\n")) for i in near_ids]    
    len_score = [abs(l - answer_len - 0.3) for l in lenth]
    print("######################### answer_len:{}, lenth:{}, len_score:{}".format(answer_len, lenth, len_score))
    #print("######################### len_score:{}".format(len_score))
    sorted_ids = [x for _, x in sorted(zip(len_score, near_ids))]
    #print("######################### sorted_ids:{}".format(sorted_ids))
    
    assert 1==0
    entry = entry['entry']
    entry['ctxs'] = sorted_ids[:num_ice]

    entry['ctxs_candidates'] = [[i] for i in sorted_ids[:num_candidates]]

    return entry


def long_knn(entry, num_candidates=1, num_ice=1):
    #print("######################### entry:{}".format(entry))
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, 16)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids
    
    answer_len = len(entry['metadata']['answer'].split("\n"))
    #print("######################### entry['metadata']['answer']:{}".format(entry['metadata']['answer']))
    #print("######################### answer_len:{}".format(answer_len))
    
    with open('./index_data/gsm8k/index_dataset.json', 'r', encoding='latin-1') as file:
        data = json.load(file)
    
    #print("######################### data[0]['answer']:{}".format(data[0]['answer']))
    lenth = [len(data[i]['answer'].split("\n")) for i in near_ids]    

    #print("######################### answer_len:{}, lenth:{}, len_score:{}".format(answer_len, lenth, len_score))
    #print("######################### len_score:{}".format(len_score))
    sorted_ids = [x for _, x in sorted(zip(lenth, near_ids), reverse=True)]
    #print("######################### sorted_ids:{}".format(sorted_ids))
    
    #assert 1==0
    entry = entry['entry']
    entry['ctxs'] = sorted_ids[:num_ice]

    entry['ctxs_candidates'] = [[i] for i in sorted_ids[:num_candidates]]

    return entry

@hydra.main(config_path="configs", config_name="iid_qa_inferencer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    if cfg.model_config.model_type == 'hf':
        accelerator = Accelerator()
        inferencer = Inferencer(cfg, accelerator)
        inferencer.forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_results()
    else:
        inferencer = APInferencer(cfg)
        inferencer.forward()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
