import argparse
import json
import math
import re
import random
random.seed(42)
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

string_to_int = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}

MMI_int_str = [str(v) for v in string_to_int.values()]

def calculate_rmse(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError("The length of labels and predictions must be the same.")
    
    mse = sum((label - pred)**2 for label, pred in zip(labels, predictions)) / len(labels)
    rmse = math.sqrt(mse)
    return rmse


def normalized_rmse_score(labels, predictions):
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    y_range = np.max(labels) - np.min(labels)
    return 1 - (rmse / y_range)


def convert_string_to_int(MMI):
    ret = []
    ret_num = []
    word_list = MMI.split(" ")
    if not (any([word in string_to_int for word in word_list]) or any([word in MMI_int_str for word in word_list])):
        word_list = MMI.split("-")
        if not (any([word in string_to_int for word in word_list]) or any([word in MMI_int_str for word in word_list])):
            word_list = MMI.split("–")
    for word in word_list:
        if word in string_to_int:
            ret.append(string_to_int[word])
        if word in MMI_int_str:
            ret_num.append(int(word))

    if ret != []:
        # if multiple MMI find in given string, average them to be the final MMI
        return sum(ret)/len(ret)
    elif ret_num != []:
        # if multiple MMI appear as number in given string, average them to be the final MMI
        return sum(ret_num)/len(ret_num)
    else:
        # if no valid MMI in given string, randomly generate one
        random_MMI = random.sample(list(string_to_int.values()),1)[0]
        return random_MMI

def extract_MMI(ret_raw):
    word_list = re.split(r'[ \n]+', ret_raw)
    ret = []
    for word in word_list:
        word = word.replace("*","")
        if word == "":
            continue
        if word[-1] in [".", ":", ","]:
            word = word[:-1]
        if word.strip() in string_to_int:
            ret.append(word.strip())
    ret = list(set(ret))
    if len(ret) == 0:
        # if no valid MMI in given string, randomly generate one
        random_MMI = random.sample(list(string_to_int.values()),1)[0]
        return random_MMI
    else:
        # if multiple MMI find in given string, average them to be the final MMI
        ret = [string_to_int[item] for item in ret]
        return sum(ret)/len(ret)

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

    if "/" in args.model:
        args.model = args.model.split("/")[1]

    if args.ICL:
        result = json.load(open("result/{}_{}_{}_ICL_{}.json".format(args.dataset, args.setting, args.model, args.k)))
    elif args.RAG:
        result = json.load(open("result/{}_{}_{}_RAG_{}.json".format(args.dataset, args.setting, args.model, args.k)))
    else:
        result = json.load(open("result/{}_{}_{}.json".format(args.dataset, args.setting, args.model)))

    zip2county = {}
    county = pd.read_csv("data/{}_DYFI_county.csv".format(args.dataset))
    for _, item in county.iterrows():
        if pd.isna(item["County"]):
            continue
        zip2county[str(item["Zip Code"])] = item["County"]

    # aggregate in zipcode-level
    result_dict = {}
    for item in result:
        zip_code = item["file_path"].split("/")[-1].split("_")[0]
        if zip_code not in result_dict:
            MMI = convert_string_to_int(item["MMI"])
            result_dict[zip_code] = {
                "MMI_predicted": [],
                "MMI": MMI
            }
        
        MMI = convert_string_to_int(item["MMI"])
        assert MMI == result_dict[zip_code]["MMI"]

        if item["MMI_predicted"] != "":
            if isinstance(item["MMI_predicted"], int) or isinstance(item["MMI_predicted"], float):
                MMI_predicted = item["MMI_predicted"]
            else:
                try:
                    MMI_predicted = float(item["MMI_predicted"])
                except:
                    MMI_predicted = convert_string_to_int(item["MMI_predicted"])
        else:
            MMI_predicted = extract_MMI(item["ret_raw"])
        result_dict[zip_code]["MMI_predicted"].append(MMI_predicted)

    for zip_code, item in result_dict.items():
        assert len(item["MMI_predicted"]) != 1
        item["MMI_predicted"] = sum(item["MMI_predicted"])/len(item["MMI_predicted"])
        result_dict[zip_code] = item

    # aggregate in county-level
    result_dict_county = {}
    for zip_code, item in result_dict.items():
        if zip_code not in zip2county:
            continue
        county = zip2county[zip_code]
        if county not in result_dict_county:
            result_dict_county[county] = {}
            result_dict_county[county]["MMI"] = []
            result_dict_county[county]["MMI_predicted"] = []
        result_dict_county[county]["MMI"].append(item["MMI"])
        result_dict_county[county]["MMI_predicted"].append(item["MMI_predicted"])

    # rmse
    labels = [item["MMI"] for item in result_dict.values()]
    predict = [item["MMI_predicted"] for item in result_dict.values()]
    rmse = calculate_rmse(labels, predict)
    labels_county = [sum(item["MMI"])/len(item["MMI"]) for item in result_dict_county.values()]
    predict_county = [sum(item["MMI_predicted"])/len(item["MMI_predicted"]) for item in result_dict_county.values()]
    rmse_county = calculate_rmse(labels_county, predict_county)

    # pearson
    pearson, _ = pearsonr(labels, predict)
    pearson_county, _ = pearsonr(labels_county, predict_county)


    print("===========================")
    print("dataset: ", args.dataset)
    print("model: ", args.model)
    print("setting: ", args.setting)
    print("ICL: ", args.ICL)
    print("RAG: ", args.RAG)
    print("k: ", args.k)
    print("zip code-level RMSE: ", rmse)
    print("county code-level RMSE: ", rmse_county)
    print("zip code-level PearsonR: ", pearson)
    print("county code-level PearsonR: ", pearson_county)
    print("===========================")



if __name__ == "__main__":
    main()