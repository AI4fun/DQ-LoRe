from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *

import re

field_getter = App()

def add_numbering(text):
    sentences = text.split('\n')  # ʹ�û��з��ָ����
    numbered_text = ''

    for i, sentence in enumerate(sentences, 1):
        numbered_text += f'{i}. {sentence}\n'  # ��ÿ������ǰ�������

    return numbered_text

def remove_brackets_content(text):
    pattern = r"<<.*?>>"  
    result = re.sub(pattern, "", text)  
    return result

@field_getter.add("q")
def get_q(entry):
    options = ""
    for op in entry['options']:
        options += op
    #return "Question: " + entry['question'] + "\nOptions: " + options
    return entry['question'] + "\nOptions: " + options


@field_getter.add("a")
def get_a(entry):
    return entry['rationale']
    #return entry['correct']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\nA: Let's think step by step.\n{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a(entry):
    #return "{ice_prompt}{question}, You need to output a json format where the thought tag is the reasoning process and the result tag is the final result, which is represented only by numbers\t".format(ice_prompt="{ice_prompt}", question=get_q(entry))
    return "{ice_prompt}{question}\t".format(ice_prompt="{ice_prompt}", question=get_q(entry))

@field_getter.add("complex_qa")
def get_complex_qa(entry):
    ans = get_a(entry)
    #ans = remove_brackets_content(ans)
    pattern = r"####\s*(-?\d+)"
    replacement = r"The answer is \1"
    new_ans = re.sub(pattern, replacement, ans)
    return "Question: {question}\nA: Let's think step by step.\n{answer}".format(question = get_q(entry), answer = new_ans)
    #return "Question:{question}\n Answer:{answer}".format(question = get_q(entry), answer = get_a(entry))
    
@field_getter.add("number_qa")
def get_number_qa(entry):
    ans = get_a(entry)
    #ans = remove_brackets_content(ans)
    pattern = r"####\s*(-?\d+)"
    replacement = r"The answer is \1"
    new_ans = re.sub(pattern, replacement, ans)
    return "Question: {question}\nA: Let's think step by step.\n{answer}".format(question = get_q(entry), answer = add_numbering(new_ans))
    #return "Question:{question}\n Answer:{answer}".format(question = get_q(entry), answer = get_a(entry))


class DatasetWrapper(ABC):
    name = "aqua"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "answer"
    hf_dataset = "/home/xiongj/icl/aqua/index_data/aqua/aqua"
    #hf_dataset_name = "main"
    field_getter = field_getter