import argparse
import pandas as pd
from tqdm import tqdm
from utils import *
from multiprocessing import Pool, cpu_count
import time
import random
random.seed(42)

def process_item(args):
    """Process a single dataset item to generate responses."""
    item, model, setting, demonstration_str_all = args
    file_path, system_prompt, earthquake_prompt, MMI = item["file_path"], item["system_prompt"], item["earthquake_prompt"], item["MMI"]
    file_path = "data/{}".format(file_path)
    if demonstration_str_all != None:
        messages = build_message(model, setting, system_prompt, earthquake_prompt+demonstration_str_all, file_path)
    else:
        messages = build_message(model, setting, system_prompt, earthquake_prompt, file_path)

    try:
        api_function = model2function[model]
        ret = api_function(model=model, messages=messages)
        ret_dict = extract_dict(ret)
        return {
            "file_path": file_path,
            "earthquake_prompt": earthquake_prompt,
            "MMI": MMI,
            "ret_raw": ret,
            "MMI_predicted": ret_dict["MMI"] if ret_dict is not None else "",
            "reasoning": ret_dict["Reasoning"] if ret_dict is not None else ""
        }
    except Exception as E:
        print(E)
        time.sleep(2)
        return process_item(args)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='2019_ridgecrest')
    argparser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo')
    argparser.add_argument('--setting', type=str, default='B+G+B+C+V')
    argparser.add_argument('--ICL', action="store_true")
    argparser.add_argument('--RAG', action="store_true")
    argparser.add_argument('--k', type=int, default='1')
    argparser.add_argument('--proc_num', type=int, default=10)
    args = argparser.parse_args()

    if args.setting == "B+G+B+C+V":
        dataset = pd.read_csv("data/{}_samples_prompt.csv".format(args.dataset))
    else:
        dataset = pd.read_csv("data/{}_samples_prompt_{}.csv".format(args.dataset, args.setting))

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

    ret_all = []

    # Prepare arguments for workers
    worker_args = [(item, args.model, args.setting, demonstration_str) for (_, item), demonstration_str in zip(dataset.iterrows(), demonstration_str_all)]

    # Use multiprocessing to process the dataset
    with Pool(processes=args.proc_num) as pool:
        ret_all = list(tqdm(pool.imap(process_item, worker_args), total=len(dataset)))
    
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