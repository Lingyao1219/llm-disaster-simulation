#!/usr/bin/env bash
#SBATCH -t 0-48:00:00
#SBATCH -c 5
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

for DATASET in 2014_napa 2019_ridgecrest
    do
        for SETTING in B+B+C+V B+G+C+V B+G+B+V B+G+B+C
            do
                for MODEL in claude-3-5-haiku-20241022
                    do
                        python main_batch_claude.py --model $MODEL --setting $SETTING --dataset $DATASET
                    done
            done
    done