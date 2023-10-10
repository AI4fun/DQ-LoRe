from src.models.api_client import run_api
from src.utils.misc import parallel_run, save_json
from omegaconf import OmegaConf
import json

import hydra
import hydra.utils as hu


@hydra.main(config_path="configs", config_name="qa_inferencer")
def main(cfg):
    model = hu.instantiate(cfg.model_config.model)
    generation_kwargs = OmegaConf.to_object(cfg.model_config.generation_kwargs)
    with open('Your dataset', 'r') as file:
        content = file.read()
    data = eval(content)

    # You can reformat your dataset here
    #
    #
    #

    with open("cot.txt") as file:
        cot_prompt = file.read()

    prompt = [cot_prompt + "\n\n" + entry['question'] for entry in data]

    responses = parallel_run(run_api, args_list=prompt,
                                 n_processes=16,
                                 client=model,
                                 **generation_kwargs)
    with open("Your path", "w") as f:
        json.dump(responses, f)


if __name__ == "__main__":
    main()
