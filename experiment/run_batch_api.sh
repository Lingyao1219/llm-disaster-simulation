#!/usr/bin/env bash
#SBATCH -t 0-48:00:00
#SBATCH -c 5
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu


for MODEL in gpt-4.1-mini-2025-04-14 gpt-4o-2024-08-06
    do
        python main_batch.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa
    done

for MODEL in claude-3-5-haiku-20241022
    do
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa
    done


python main_batch_claude_data_leakage.py --model claude-3-5-haiku-20241022 --setting data_leakage_test --dataset 2014_napa