import json
import time
import argparse
from anthropic import Anthropic
from typing import List, Dict
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from utils import *
import pandas as pd
import random
random.seed(42)


# Initialize the Anthropic client with your API key
client = Anthropic(api_key='Your API key')

def chunked_requests(requests, chunk_size):
    for i in range(0, len(requests), chunk_size):
        yield requests[i:i + chunk_size]

def create_batch_requests(mode: str, dataset: List[Dict], model: str, demonstration_str_all: List) -> List[Request]:
    """
    Create batch requests from instruction data.
    """
    batch_requests = []
    for (i, item), demonstration_str in zip(dataset.iterrows(),demonstration_str_all):
        file_path, system_prompt, earthquake_prompt, MMI = item["file_path"], item["system_prompt"], item["earthquake_prompt"], item["MMI"]
        file_path = "data/{}".format(file_path)
        if demonstration_str != None:
            messages = build_message(model, mode, system_prompt, earthquake_prompt+demonstration_str, file_path)
        else:
            messages = build_message(model, mode, system_prompt, earthquake_prompt, file_path)
        batch_request = Request(
            custom_id=str(i),
            params=MessageCreateParamsNonStreaming(
                model=model,
                temperature=0.0,
                max_tokens=4096,
                **messages
            )
        )
        batch_requests.append(batch_request)
    return batch_requests

def check_batch_status(batch_id: str):
    """
    Check the status of a batch job.
    """
    return client.messages.batches.retrieve(batch_id)

def process_batch_results(batch_id: str, dataset: List[Dict]) -> List[Dict]:
    """
    Process batch results and extract instruction-output pairs.
    
    Args:
        batch_id: ID of the batch job
        dataset: Original list of instruction dictionaries
        
    Returns:
        List of dictionaries containing instructions and outputs
    """
    synthetic_response = []
    
    # Create a dictionary to store original instructions
    instruction_dict = {}
    for i, item in dataset.iterrows():
        instruction_dict[str(i)] = item["earthquake_prompt"]
    
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            # Get the instruction using custom_id
            instruction = instruction_dict[result.custom_id]
            output = result.result.message.content[0].text
            synthetic_response.append({
                "instruction": instruction,
                "output": output
            })
        else:
            print(f"Request {result.custom_id} failed with status: {result.result.type}")
    
    return synthetic_response

def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='2014_napa')
    argparser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022')
    argparser.add_argument('--setting', type=str, default='B+G+B+C+V')
    argparser.add_argument('--ICL', action="store_true")
    argparser.add_argument('--RAG', action="store_true")
    argparser.add_argument('--k', type=int, default='1')
    args = argparser.parse_args()

    # Step 1: Load instruction data
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

    batch_requests = create_batch_requests(args.setting, dataset, args.model, demonstration_str_all)

    synthetic_response = []
    for idx, batch_chunk in enumerate(chunked_requests(batch_requests, chunk_size=500)):
        print(f"Submitting chunk {idx + 1} with {len(batch_chunk)} requests...")
        message_batch = client.messages.batches.create(requests=batch_chunk)
        print(f"Batch job created. Batch ID: {message_batch.id}")

        # Polling
        while True:
            status = check_batch_status(message_batch.id)
            print(f"Current status: {status.processing_status}")
            print(f"Request counts: {status.request_counts}")
            if status.processing_status == "ended":
                print("Batch job completed.")
                break
            time.sleep(30)

        # Process and collect results from this chunk
        responses = process_batch_results(message_batch.id, dataset)
        synthetic_response.extend(responses)



    # Step 2: extract prediction from synthetic response
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