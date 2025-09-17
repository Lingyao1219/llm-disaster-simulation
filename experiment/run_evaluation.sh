
for DATASET in 2014_napa 2019_ridgecrest
    do
        for MODEL in claude-3-5-haiku-20241022 gpt-4.1-mini-2025-04-14 gpt-4o-2024-08-06 Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-32B-Instruct Qwen/Qwen2.5-VL-72B-Instruct meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo
            do
                python evaluation.py --model $MODEL --dataset $DATASET
            done
    done


for DATASET in 2014_napa 2019_ridgecrest
    do
        for MODEL in claude-3-5-haiku-20241022 Qwen/Qwen2.5-VL-7B-Instruct
            do
                for K in 1 3 5
                    do
                        python evaluation.py --model $MODEL --dataset $DATASET --ICL --k $K
                        python evaluation.py --model $MODEL --dataset $DATASET --RAG --k $K
                    done
            done
    done


for DATASET in 2014_napa 2019_ridgecrest
    do
        for MODEL in claude-3-5-haiku-20241022 Qwen/Qwen2.5-VL-7B-Instruct
            do
                for SETTING in B+B+C+V B+G+C+V B+G+B+V B+G+B+C
                    do
                        python evaluation.py --model $MODEL --dataset $DATASET --setting $SETTING
                    done
            done
    done


for DATASET in 2014_napa
    do
        for MODEL in claude-3-5-haiku-20241022 gpt-4.1-mini-2025-04-14
            do
                for SETTING in data_leakage_test
                    do
                        python evaluation.py --model $MODEL --dataset $DATASET --setting $SETTING
                    done
            done
    done