import warnings
import logging
import hydra
import hydra.utils as hu
import numpy as np
from datasets import load_metric
from transformers import Trainer, EvalPrediction, EarlyStoppingCallback, set_seed
from src.utils.collators import IBNDataCollatorWithPadding
from src.models.biencoder import BiEncoder
import torch

logger = logging.getLogger(__name__)
# 初始化 CUDA
torch.cuda.init()

# # 初始化 cuBLAS
# torch.cuda.blas.init()


class RetrieverTrainer:

    def __init__(self, cfg) -> None:
        self.training_args = hu.instantiate(cfg.training_args)
        #self.dataset_reader = hu.instantiate(cfg.dataset_reader)        
        self.index_reader = hu.instantiate(cfg.index_reader)
        encoded_index = list(self.index_reader)

        #train_dataset, eval_dataset = self.dataset_reader.split_dataset(test_size=0.1, seed=42)
        #logger.info(f"train size: {len(train_dataset)}, eval size: {len(eval_dataset)}")
        
        # qa_dataset_reader
        self.qa_dataset_reader = hu.instantiate(cfg.qa_dataset_reader)
        qa_train_dataset, qa_eval_dataset = self.qa_dataset_reader.split_dataset(test_size=0.1, seed=42)
        logger.info(f"qa_train size: {len(qa_train_dataset)}, eval size: {len(qa_eval_dataset)}")
        

        model_config = hu.instantiate(cfg.model_config)
        '''
        if cfg.pretrained_model_path is not None:
            self.model = BiEncoder.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = BiEncoder(model_config)
        '''
            
        self.qa_model = BiEncoder(model_config)    # qa_model config = model_config
        
        
        '''
        data_collator = IBNDataCollatorWithPadding(tokenizer=self.dataset_reader.tokenizer,
                                                   encoded_index=encoded_index,
                                                   **cfg.collector)
        '''

        qa_data_collator = IBNDataCollatorWithPadding(tokenizer=self.qa_dataset_reader.tokenizer,
                                                   encoded_index=encoded_index,
                                                   **cfg.collector)

        self.metric = load_metric('src/metrics/accuracy.py')
        '''
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.dataset_reader.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        '''
        
        
        self.qa_trainer = Trainer(
            model=self.qa_model,
            args=self.training_args,
            train_dataset=qa_train_dataset,
            eval_dataset=qa_eval_dataset,
            tokenizer=self.qa_dataset_reader.tokenizer,
            data_collator=qa_data_collator,
            compute_metrics=self.compute_metrics
        )
        
        #self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
        #self.qa_trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
    def train(self):
        #self.trainer.train()
        #print("here!!!!!!!!!!!!!!!!!!!!")
        #self.trainer.save_model(self.training_args.output_dir)
        #self.trainer.tokenizer.save_pretrained(self.training_args.output_dir)
        #del(self.trainer)
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.qa_trainer.train()
        print("qa_trainer is training...")
        self.qa_trainer.save_model('output/epr/aqua/EleutherAI/gpt-neo-2.7B/bert-fix_ctx-shared-bs64/qa_model')
        self.qa_trainer.tokenizer.save_pretrained('output/aqua/gsm8k/EleutherAI/gpt-neo-2.7B/bert-fix_ctx-shared-bs64/qa_model')

    def compute_metrics(self, p: EvalPrediction):
        predictions = np.argmax(p.predictions, axis=1)  # p.predictions is logits, so we must take argmax
        return self.metric.compute(predictions=predictions, references=p.label_ids)

import os
@hydra.main(config_path="configs", config_name="retriever_trainer")
def main(cfg):
    os.environ["WANDB_MODE"] = "offline"
    logger.info(cfg)
    set_seed(43)

    trainer = RetrieverTrainer(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.training_args.do_train:
            print("11111111111111111111111111111111")
            trainer.train()
        if cfg.training_args.do_eval:
            # logger.info(trainer.trainer.evaluate(eval_dataset=trainer.trainer.train_dataset, metric_key_prefix="train"))
            logger.info(trainer.qa_trainer.evaluate())


if __name__ == "__main__":
    main()
