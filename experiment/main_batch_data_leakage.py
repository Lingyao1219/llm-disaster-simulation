import time
import json
from openai import OpenAI
import argparse
from copy import deepcopy
import os
os.environ["HF_HOME"]="/scratch/daweili5/hf_cache"
from datasets import load_dataset
from utils import *
import pandas as pd
import anthropic
import random
random.seed(42)

# Initialize the OpenAI client with your API key
client = OpenAI(api_key='Your API key')

def build_chunk(mode, dataset, chunk_size, model, demonstration_str_all):
    batch_data_all = []
    for (i, item), demonstration_str in zip(dataset.iterrows(),demonstration_str_all):
        file_path, system_prompt, earthquake_prompt, MMI = item["file_path"], item["system_prompt"], item["earthquake_prompt"], item["MMI"]
        file_path = "data/{}".format(file_path)
        if demonstration_str != None:
            messages = build_message(model, mode, system_prompt, earthquake_prompt+demonstration_str, file_path)
        else:
            messages = build_message(model, mode, system_prompt, earthquake_prompt, file_path)
        batch_template = {
            "custom_id": str(i), 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": model, 
                "messages": messages,
                "temperature": 0.0
            }
        }
        batch_data_all.append(batch_template)

    file_path_list = []
    chunk_num = int(len(dataset)/chunk_size+1)
    for i in range(chunk_num):
        if i == chunk_num-1:
            batch_data_chunk = batch_data_all[i*chunk_size:]
        else:
            batch_data_chunk = batch_data_all[i*chunk_size:(i+1)*chunk_size]

        if batch_data_chunk == []:
            continue

        input_file_path = "data/{}_{}_part{}_input.jsonl".format(mode, model, str(i))
        output_file_path = "data/{}_{}_part{}_output.json".format(mode, model, str(i))
        file_path_list.append((input_file_path, output_file_path))
        with open(input_file_path, "w") as f:
            for item in batch_data_chunk:
                f.write(json.dumps(item)+'\n')

    return file_path_list

def upload_batch_input_file(file_path):
    # Upload the batch input file (in JSONL format)
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    return batch_input_file.id

def create_batch_job(batch_input_file_id):
    # Create a batch job for processing the input file
    ret = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "teacher bias"}
    )
    return ret.id

def check_batch_status(batch_id):
    # Check the status of the batch job
    return client.batches.retrieve(batch_id)

def retrieve_batch_results(output_file_id, result_file_path):
    # Retrieve and save the results of the batch job
    ret_json = []
    file_response = client.files.content(output_file_id)

    # Parse the response line by line and load it into JSON format
    for line in file_response.text.strip().split('\n'):
        ret_json.append(json.loads(line))

    # Write the results to a file
    with open(result_file_path, 'w') as f:
        json.dump(ret_json, f, indent=2)

def extract_result(input_file_path, output_file_path):
    input_file = []
    with open(input_file_path) as f:
        for line in f.readlines():
            input_file.append(json.loads(line))
    output_file = json.load(open(output_file_path))

    synthetic_response_batch = []
    for item_input, item_output in zip(input_file, output_file):
        assert item_input["custom_id"] == item_output["custom_id"]
        instruction = item_input["body"]["messages"][1]["content"][0]["text"]
        output = item_output["response"]["body"]["choices"][0]["message"]["content"]
        synthetic_response_batch.append({
            "instruction": instruction,
            "output": output
        })

    return synthetic_response_batch

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='2019_ridgecrest')
    argparser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022')
    argparser.add_argument('--setting', type=str, default='B+G+B+C+V')
    argparser.add_argument('--chunk_size', type=int, default=1000)
    argparser.add_argument('--ICL', action="store_true")
    argparser.add_argument('--RAG', action="store_true")
    argparser.add_argument('--k', type=int, default='1')
    args = argparser.parse_args()

    # Step 0: Chunk input file
    if args.setting == "B+G+B+C+V":
        dataset = pd.read_csv("data/{}_samples_prompt.csv".format(args.dataset))
    else:
        dataset = pd.read_csv("data/{}_samples_prompt_{}.csv".format(args.dataset, args.setting))


    dataset_org = pd.read_csv("data/{}_samples_prompt.csv".format(args.dataset))
    prompt2file, prompt2mmi = {}, {}
    for _, item in dataset_org.iterrows():
        earthquake_prompt, MMI, file_path = item["location_id"], item["MMI"], item["file_path"]
        prompt2file[earthquake_prompt] = file_path
        prompt2mmi[earthquake_prompt] = MMI

    file_path_all, mmi_all, clean_prompt_all = [], [], []
    for _, item in dataset.iterrows():
        earthquake_prompt = item["location_id"]
        clean_prompt_all.append(item["clean_prompt"])
        file_path_all.append(prompt2file[earthquake_prompt])
        mmi_all.append(prompt2mmi[earthquake_prompt])

    dataset["file_path"] = file_path_all
    dataset["MMI"] = mmi_all
    dataset["earchquake_prompt"] = clean_prompt_all


    demonstration_str_all = [None for _ in range(len(dataset))]

    if args.ICL:
        ICL_samples = pd.read_csv("data/{}_rag_samples_prompt.csv".format(args.dataset))
        ICL_samples = [item for _, item in ICL_samples.iterrows()]
        demonstration = random.sample(ICL_samples, args.k)
        demonstration_str = "Here are {} demonstrations for reference:\n\n".format(args.k)
        for item in demonstration:
            demonstration = item["earthquake_prompt"].split("Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale.")[0]
            demonstration_str += demonstration + "\n" + "MMI: {}".format(item["MMI"]) + "\n\n"
        demonstration_str_all = [demonstration_str] * len(dataset)

    if args.RAG:
        import ast
        demonstration_str_all = []
        dataset_rag = dataset = pd.read_csv("data/{}_samples_prompt_rag_{}.csv".format(args.dataset, args.k))
        for _, item in dataset_rag.iterrows():
            retrieval_result = item["retrieval result"]
            demonstration_str = "Here are {} demonstrations for reference:\n\n".format(args.k)
            retrieval_result = ast.literal_eval(retrieval_result)
            for earthquake_prompt, MMI in retrieval_result:
                demonstration = earthquake_prompt.split("Based on the information provided, ASSESS the potential earthquake damage level using the Modified Mercalli Intensity (MMI) scale.")[0]
                demonstration_str += demonstration + "\n" + "MMI: {}".format(MMI) + "\n\n"
            demonstration_str_all.append(demonstration_str)


    file_path_list = build_chunk(args.setting, dataset, args.chunk_size, args.model, demonstration_str_all)

    synthetic_response = []
    for input_file_path, output_file_path in file_path_list:
        # Step 1: Upload the batch input file
        batch_input_file_id = upload_batch_input_file(input_file_path)
        print(f"Batch input file uploaded. Batch File ID: {batch_input_file_id}")

        # Step 2: Create the batch job
        batch_id = create_batch_job(batch_input_file_id)
        print(f"Batch job created. Batch ID: {batch_id}")

        # Step 3: Poll the status every 30 seconds until completion
        while True:
            status = check_batch_status(batch_id)
            print(f"Current status: {status.status}")
            
            if status.status == 'completed':
                print("Batch job completed.")
                break
            elif status.status == 'failed':
                print("Batch job failed.")
                print(status)
                return
            
            time.sleep(30)  # Wait for 30 seconds before checking again

        # Step 4: Retrieve and save the results
        output_file_id = status.output_file_id
        retrieve_batch_results(output_file_id, output_file_path)
        print(f"Results saved to {output_file_path}")

        # Step 5: Extract output from each results file
        synthetic_response_batch = extract_result(input_file_path, output_file_path)
        synthetic_response.extend(synthetic_response_batch)

    # Step 6: extract prediction from synthetic response
    ret_all = []
    for response, (i, item) in zip(synthetic_response, dataset.iterrows()):
        instruction, ret = response["instruction"], response["output"]
        file_path, system_prompt, earthquake_prompt, MMI = item["file_path"], item["system_prompt"], item["earthquake_prompt"], item["MMI"]
        assert instruction == earthquake_prompt
        try:
            ret_dict = extract_dict(ret)
            ret_all.append({
                "file_path": file_path,
                "earthquake_prompt": earthquake_prompt,
                "MMI": MMI,
                "ret_raw": ret,
                "MMI_predicted": ret_dict["MMI"] if ret_dict is not None else "",
                "reasoning": ret_dict["Reasoning"] if ret_dict is not None else ""
            })
        except Exception as E:
            print(E)
            continue

    if "/" in args.model:
        args.model = args.model.split("/")[1]

    if args.ICL:
        with open("result/{}_{}_{}_ICL_{}.json".format(args.dataset, args.setting, args.model, args.k), "w") as f:
            json.dump(ret_all, f, indent=2)
    elif args.RAG:
        with open("result/{}_{}_{}_RAG_{}.json".format(args.dataset, args.setting, args.model, args.k), "w") as f:
            json.dump(ret_all, f, indent=2)
    else:
        with open("result/{}_{}_{}.json".format(args.dataset, args.setting, args.model), "w") as f:
            json.dump(ret_all, f, indent=2)

if __name__ == "__main__":
    main()