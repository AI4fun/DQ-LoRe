#!/usr/bin/python3
# -*- coding: utf-8 -*-
import openai
import time
import random
import numpy as np
import logging
import codecs
import os

logger = logging.getLogger(__name__)


class OpenAIClient():
    def __init__(self, keys_file):
        if os.path.exists(keys_file):
            with open(keys_file) as f:
                self.keys = [i.strip() for i in f.readlines()]
        else:
            self.keys = [os.environ['OPENAI_TOKEN']]
        self.n_processes = len(self.keys)

    def call_api(self, prompt: str, engine: str, max_tokens=2000, temperature=1,
                 stop=None, n=None, echo=False):
        result = None
        if temperature == 0:
            n = 1

        while result is None:
            try:
                key = random.choice(self.keys)
                
                result = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    api_key=key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                    #stop=stop,
                    logprobs=1,
                    echo=echo
                )
                '''
                result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    api_key=key,
                    max_tokens=3000,
                    temperature=0.7,
                    n=8,
                    messages=[
                    {"role": "user", "content": prompt}
                    ]
                )
                '''
                time.sleep(5)
                print("=============call api result:{}".format(result))
                return result
            except Exception as e:
                logger.info(f"{str(e)}, 'Retry.")
                time.sleep(5)

    def extract_response(self, response):
        texts = [r['text'] for r in response['choices']]
        logprobs = [np.mean(r['logprobs']['token_logprobs']) for r in response['choices']]
        return [{"text": text, "logprob": logprob} for text, logprob in zip(texts, logprobs)]

    def extract_loss(self, response):
        lens = len(response['choices'][0]['logprobs']['tokens'])
        ce_loss = -sum(response['choices'][0]['logprobs']['token_logprobs'][1:])
        return ce_loss / (lens-1)  # no logprob on first token
    
    def extract_scorer_response(self, response):
        texts = [r['text'] for r in response['choices']]
        logprobs = [np.mean(r['logprobs']['token_logprobs']) for r in response['choices']]
        return [{"text": text, "logprob": logprob} for text, logprob in zip(texts, logprobs)]


def run_scorer_api(args, **kwargs):
    if isinstance(args, tuple):
        prompt, choices = args
    else:
        prompt, choices = args, None
    client = kwargs.pop('client')
    if choices is None:
        response = client.call_api(prompt=prompt, **kwargs)
        response = client.extract_scorer_response(response)
    else:
        kwargs.update({"echo": True, "max_tokens": 0})
        losses = np.array([client.extract_loss(client.call_api(prompt=prompt+choice, **kwargs))
                          for choice in choices])
        pred = int(losses.argmin())  # numpy int64 can't been saved by json, convert to int
        # hard code, mimic normal response
        response = [{'text': pred}]
    return response

def run_api(args, **kwargs):
    if isinstance(args, tuple):
        prompt, choices = args
    else:
        prompt, choices = args, None
    client = kwargs.pop('client')
    if choices is None:
        response = client.call_api(prompt=prompt, **kwargs)
        response = client.extract_response(response)
    else:
        kwargs.update({"echo": True, "max_tokens": 0})
        losses = np.array([client.extract_loss(client.call_api(prompt=prompt+choice, **kwargs))
                          for choice in choices])
        pred = int(losses.argmin())  # numpy int64 can't been saved by json, convert to int
        # hard code, mimic normal response
        response = [{'text': pred}]
    return response
