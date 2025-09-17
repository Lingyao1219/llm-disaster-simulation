#!/usr/bin/env bash
#SBATCH -t 0-48:00:00
#SBATCH -c 5
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu



for MODEL in claude-3-5-haiku-20241022
    do
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2019_ridgecrest --k 1 --RAG
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2019_ridgecrest --k 3 --RAG
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2019_ridgecrest --k 5 --RAG

        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa --k 1 --RAG
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa --k 3 --RAG
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa --k 5 --RAG


        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2019_ridgecrest --k 1 --ICL
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2019_ridgecrest --k 3 --ICL
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2019_ridgecrest --k 5 --ICL

        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa --k 1 --ICL
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa --k 3 --ICL
        python main_batch_claude.py --model $MODEL --setting B+G+B+C+V --dataset 2014_napa --k 5 --ICL
    done