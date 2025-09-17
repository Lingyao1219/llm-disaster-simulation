#!/usr/bin/env bash
#SBATCH -t 0-3:00:00
#SBATCH --gres=gpu:a100:5
#SBATCH -c 20
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daweili5@asu.edu

nvidia-smi

export HF_HOME=/scratch/daweili5/hf_cache
export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ..`

GPU=0,1,2,3
PORT=8001
TP_SIZE=4

export CUDA_VISIBLE_DEVICES=${GPU}
export base_url=http://localhost:${PORT}/v1

declare -A MODEL_ZOO
MODEL_ZOO["Qwen2.5-VL-3B-Instruct"]="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_ZOO["Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ZOO["Qwen2.5-VL-32B-Instruct"]="Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_ZOO["Qwen2.5-VL-72B-Instruct"]="Qwen/Qwen2.5-VL-72B-Instruct"


# for MODEL in Qwen2.5-VL-3B-Instruct Qwen2.5-VL-7B-Instruct Qwen2.5-VL-32B-Instruct
# for MODEL in Qwen2.5-VL-3B-Instruct
# for MODEL in Qwen2.5-VL-7B-Instruct

for SETTING in B+G+B+C+V
       do
              for MODEL in Qwen2.5-VL-3B-Instruct Qwen2.5-VL-7B-Instruct Qwen2.5-VL-32B-Instruct Qwen2.5-VL-72B-Instruct

                     do

                            model_name=${MODEL_ZOO["$MODEL"]}

                            # run the vllm server
                            python -m vllm.entrypoints.openai.api_server \
                                   --model ${model_name} \
                                   --tensor-parallel-size ${TP_SIZE} \
                                   --download-dir "${HOME_DIR}/model_cache/" \
                                   --port ${PORT} &
                            
                            PID=$!
                            sleep 600

                            for DATASET in 2019_ridgecrest

                                   do

                                          # run sampling
                                          python main.py --model ${model_name} \
                                                 --setting $SETTING \
                                                 --dataset $DATASET


                                   done
                            
                            kill $PID
                     done
       done