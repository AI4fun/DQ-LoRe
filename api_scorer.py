import glob
import json
import os
import logging
import hydra
import torch
import tqdm
from transformers import set_seed
from accelerate import Accelerator
from inferencer import Inferencer, APInferencer
from src.utils.misc import save_json

# Inferencer.py import 
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
from src.models.api_client import run_api, run_scorer_api
from src.utils.misc import parallel_run, save_json
from src.models.model import ppl_generate


import openai
import time
import random
import numpy as np
import logging
import codecs
import os


logger = logging.getLogger(__name__)


class Scorer(APInferencer):

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
            
        
        
        prompts = [entry['metadata']['prompt'] for entry in dataloader]
        metadata = [entry['metadata'] for entry in dataloader]
        print("!!!!!!!!!!!!!!! prompts[0]:{}".format(prompts[0]))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! metadata[0]:{}".format(metadata[0]))
        #assert 1==0
        responses = parallel_run(run_scorer_api, args_list=prompts,
                                 n_processes=128,
                                 client=self.model,
                                 **self.generation_kwargs)
            
        for mdata, response in zip(metadata, responses):
            mdata['score'] = response[0]['logprob']
        
        with open(f"{self.output_file}tmp_{self.accelerator.device}.bin", "w") as f:
            json.dump(metadata, f)
    


    def write_results(self):
        data = []
        for i, path in enumerate(glob.glob(f"{self.output_file}tmp_*.bin")):
            with open(path) as f:
                one_device = json.load(f)
                logger.info(f"device: {i}, idx {[i['idx'] for i in one_device][:200]}...")
                data.extend(one_device)

        # grouping results by uid
        example_dict = {}
        uid_field = 'idx'
        for entry in data:
            ctxs = {"ctxs": entry.pop('ctxs'), "score": entry.pop("score")}
            if entry[uid_field] not in example_dict:
                entry['ctxs_candidates'] = [ctxs]
                example_dict[entry[uid_field]] = entry
            else:
                example_dict[entry[uid_field]]['ctxs_candidates'].append(ctxs)

        example_list = list(example_dict.values())
        mrr = 0
        num_candidates = len(example_list[0]['ctxs_candidates'])
        for entry in example_list:
            assert len(entry['ctxs_candidates']) == num_candidates, f"{len(entry['ctxs_candidates'])}!={num_candidates}"

            sorted_tuple = sorted(enumerate(entry['ctxs_candidates']), key=lambda x: x[1]['score'])
            entry['ctxs_candidates'] = [i[1]['ctxs'] for i in sorted_tuple]
            entry['ctxs'] = entry['ctxs_candidates'][0]  # set top-scored cand to ctxs
            mrr += 1/([i[0] for i in sorted_tuple].index(0)+1)
        logger.info(f"MRR: {mrr/len(example_list)}")

        save_json(self.output_file, example_list)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)


@hydra.main(config_path="configs", config_name="api-scorer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    accelerator = Accelerator()
    scorer = Scorer(cfg, accelerator)

    scorer.forward()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        scorer.write_results()


if __name__ == "__main__":
    main()
