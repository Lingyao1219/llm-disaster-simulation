#!/usr/bin/env bash
#SBATCH -t 0-72:00:00
#SBATCH -c 10
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu


for MODEL in meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo Qwen/Qwen2.5-VL-72B-Instruct
    do
        python main.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa
    done