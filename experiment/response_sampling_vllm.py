import argparse
import os
import json
# from utils import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI
import time
import re
from datasets import load_dataset

def generate_vllm(prompt, model, client, n=1):

    kwargs = {
        "model": model,
        'messages':[
            {"role": "user", "content": prompt}
        ],
        "n":n,
        "temperature":0.0
    }
    completion = client.chat.completions.create(**kwargs)
    res = [item.message.content.strip() for item in completion.choices]
    return res

def initialize_client(base_url):
    """Initialize a new OpenAI client for each process."""
    # from openai_client import OpenAI  # Import here to avoid pickling issues
    return OpenAI(api_key="EMPTY", base_url=base_url)


def process_item(args):
    """Process a single dataset item to generate responses."""
    item, model, base_url, n = args
    client = initialize_client(base_url)
    instruction = item["problem"]
    try:
        responses = generate_vllm(prompt=instruction, model=model, client=client, n=n)
        item["responses"] = responses
        return item
    except Exception as E:
        time.sleep(2)
        return process_item(args)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='UWNSL/MATH_training_split_long_cot')
    argparser.add_argument('--save_path', type=str, default='data')
    argparser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')
    argparser.add_argument('--openai_base_url', type=str, help="url to call the model", default='http://localhost:8001/v1')
    argparser.add_argument('--n', type=str, help="how many responses to sample for one prompt", default=1)
    args = argparser.parse_args()
    
    dataset = load_dataset(args.dataset,split="train")
    dataset_name = args.dataset.split("/")[0]
    dataset_syn = []


    # Prepare arguments for workers
    worker_args = [(item, args.model, args.openai_base_url, args.n) for item in dataset]
    
    # Use multiprocessing to process the dataset
    with Pool(processes=20) as pool:
        dataset_syn = list(tqdm(pool.imap(process_item, worker_args), total=len(dataset)))

   
    model_name = args.model.replace("/","_")
    with open(os.path.join(args.save_path, "{}_{}_{}.json".format(dataset_name, model_name, args.n)), "w") as f:
        json.dump(dataset_syn, f, indent=2)


if __name__ == "__main__":
    main()