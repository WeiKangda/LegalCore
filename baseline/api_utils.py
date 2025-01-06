from typing import Any
from openai import OpenAI
import openai
from tenacity import retry, wait_chain, wait_fixed
import google.generativeai as genai
# import boto3
import json
import csv
import time
from tqdm.auto import tqdm
import traceback
from collections import defaultdict
from dotenv import load_dotenv
import os
load_dotenv()

gpt_apis = os.getenv("GPT_APIS").split(",")
gemini_apis = os.getenv("GEMINI_APIS").split(",")

class GPT:
    def __init__(self, model_name, temperature=1, seed=42, api_idx=0):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=gpt_apis[api_idx],
        )
        self.T = temperature
        self.seed = seed

    def __call__(self, prompt, n: int = 1, debug=False, **kwargs: Any) -> Any:
        # prompt = [{'role': 'user', 'content': prompt}]
        if debug:
            return self.client.chat.completions.create(messages=prompt, n=n, model=self.model_name, temperature=self.T,
                                                       seed=self.seed, **kwargs)

        else:
            return self.call_wrapper(messages=prompt, n=n, model=self.model_name, temperature=self.T, seed=self.seed,
                                     **kwargs)

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(1)] +
                            [wait_fixed(5) for i in range(1)] +
                            [wait_fixed(10)]))  # not use for debug
    def call_wrapper(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        return response
    def resp_parse(self, response) -> list:
        n = len(response.choices)
        return [response.choices[i].message.content for i in range(n)]

    def eval_call_wrapper(self, prompt, **kwargs: Any) -> Any:
        retry_count = 0
        stop = False
        while retry_count < 5 or stop:
            try:
                return self.__call__(prompt, debug=True, **kwargs)

            except openai.BadRequestError as e:
                err = e.body['message']
                if 'repetitive pattern' in err:
                    print(f"error: {err}, use default fail eval")
                    stop = True
                    return None
            except Exception as e:
                print(f"error exception: {e}")
                print(kwargs)
                time.sleep(2 ** retry_count + 1)
                retry_count += 1

    # only for handle eval_cot5 error on repetitive pattern
    def eval_call(self, prompt, n: int = 1, debug=False, **kwargs: Any) -> Any:
        if debug:
            return self.__call__(prompt, debug=True, n=n, **kwargs)

        else:
            return self.eval_call_wrapper(prompt, n=n, **kwargs)

def load_model(model_name, api_idx, **kwargs):
    if "gpt" in model_name and "gpt2" not in model_name:
        return GPT(model_name, **kwargs)

    # elif "gpt2" in model_name:
    #     return GPT2(model_name, api_idx=api_idx, **kwargs)
    #
    # elif "gemini" in model_name:
    #     return Gemini(api_idx=api_idx, **kwargs)
    #
    # elif "claude" in model_name:
    #     return Claude(model_name, **kwargs)
    # elif "llama" in model_name:
    #     if "llama-vllm" in model_name:
    #         return Llama_vllm(model_name, **kwargs)
    #     return Llama_api(model_name, **kwargs)
    else:
        raise ValueError(f"model_name invalid")