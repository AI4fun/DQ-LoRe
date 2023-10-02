import torch
import logging
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from src.utils.tokenizer_util import get_tokenizer

logger = logging.getLogger(__name__)


def _encode_field(example, idx, **kwargs):
    field_getter = kwargs['field_getter']
    tokenizer = kwargs['tokenizer']
    truncation = kwargs['truncation']
    text = field_getter(example)
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ text:{}".format(text))
    #text = text + "You need to output a json format where the thought tag is the reasoning process and the result tag is the final result, which is represented only by numbers."
    tokenized_inputs = tokenizer.encode_plus(text, truncation=truncation, return_tensors='pt')
    return {
        'input_ids': tokenized_inputs.input_ids[0],
        'attention_mask': tokenized_inputs.attention_mask[0],
        "metadata": {"id": idx, "len": len(tokenized_inputs.input_ids[0]),
                     "text": text}
    }


def encode_field(tokenizer, dataset_wrapper, field, truncation):
    remove_columns = [col for col in dataset_wrapper.dataset.column_names]
    print("bbbbbbbbbbbbbbbbbbbbbbbbbb")
    print(dataset_wrapper.dataset)
    print(remove_columns)
    encoded_dataset = dataset_wrapper.dataset.map(
        _encode_field,
        load_from_cache_file=False,
        with_indices=True,
        remove_columns=remove_columns,
        fn_kwargs={'field_getter': dataset_wrapper.field_getter.functions[field],
                   'tokenizer': tokenizer, 'truncation': truncation}
    )
    print(encoded_dataset)
    return encoded_dataset


class BaseDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, model_name, field, dataset_path=None, dataset_split=None, ds_size=None) -> None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!base_dsr model_name:{}".format(model_name))
        self.tokenizer = get_tokenizer(model_name)
        self.init_dataset(task_name, field, dataset_path, dataset_split, ds_size)

    def init_dataset(self, task_name, field, dataset_path, dataset_split, ds_size, truncation=True):
        print("===task_name:{}".format(task_name))
        print("===dataset_split:{}".format(dataset_split))
        
        self.dataset_wrapper = get_dataset_wrapper(task_name,
                                                   dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)
        #assert 1==0
        self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, truncation)

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)
