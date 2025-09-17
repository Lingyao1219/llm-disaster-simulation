#!/usr/bin/env bash
#SBATCH -t 0-8:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH -c 20
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

export HF_HOME=/scratch/daweili5/hf_cache
export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ..`

GPU=0,1
PORT=8001
TP_SIZE=2

export CUDA_VISIBLE_DEVICES=${GPU}
export base_url=http://localhost:${PORT}/v1

declare -A MODEL_ZOO
MODEL_ZOO["Qwen2.5-VL-3B-Instruct"]="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_ZOO["Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ZOO["Qwen2.5-VL-32B-Instruct"]="Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_ZOO["Qwen2.5-VL-72B-Instruct"]="Qwen/Qwen2.5-VL-72B-Instruct"


for MODEL in Qwen2.5-VL-7B-Instruct
    do
        model_name=${MODEL_ZOO["$MODEL"]}

        # run the vllm server
        python -m vllm.entrypoints.openai.api_server \
            --model ${model_name} \
            --tensor-parallel-size ${TP_SIZE} \
            --download-dir "${HOME_DIR}/model_cache/" \
            --port ${PORT} &
        
        PID=$!

        sleep 300

        for K in 1 3 5
            do
                for SETTING in B+G+B+C+V
                    do
                        for DATASET in 2014_napa 2019_ridgecrest

                                do
                                        # run sampling
                                        python main.py --model ${model_name} \
                                            --setting $SETTING \
                                            --dataset $DATASET \
                                            --k $K \
                                            --ICL

                                        python main.py --model ${model_name} \
                                            --setting $SETTING \
                                            --dataset $DATASET \
                                            --k $K \
                                            --RAG

                                done
                    done
            done
        kill $PID
    done