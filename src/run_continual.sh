#!/bin/bash

# 공통 하이퍼파라미터 설정
hf_path="vit_large_patch14_dinov2.lvd142m"
image_size=336
batch_size=16
seed=0
num_workers=8

# 실행

# _except_mvtec_visa
# _except_continual_ad

JSON_PATH_BASE="base_classes_except_continual_ad"
TASK_JSON_GROUPS=(
    "5classes_tasks_except_continual_ad" \
    "10classes_tasks_except_continual_ad" \
    "30classes_tasks_except_continual_ad"
)
NUM_TASKS_GROUPS=(
    6 \
    3 \
    1
)

# BASE 학습 (task_id=0)
echo "[INFO] Start BASE PHASE"
python /workspace/MegaInspection/GeneralAD/main.py \
    --run_type general_ad \
    --hf_path $hf_path \
    --epochs 50 \
    --image_size $image_size \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --seed $seed \
    --task_id 0 \
    --json_path $JSON_PATH_BASE

# CONTINUAL 학습 (task_id=1~5)
for ((i=0; i<${#TASK_JSON_GROUPS[@]}; i++)); do
    NUM_TASKS=${NUM_TASKS_GROUPS[$i]}
    for ((TASK_ID=1; TASK_ID<=NUM_TASKS; TASK_ID++)); do
        echo "[INFO] Start CONTINUAL PHASE - Task $TASK_ID"
        python /workspace/MegaInspection/GeneralAD/main.py \
            --run_type general_ad \
            --hf_path $hf_path \
            --epochs 20 \
            --image_size $image_size \
            --batch_size $batch_size \
            --num_workers $num_workers \
            --seed $seed \
            --task_id $TASK_ID \
            --json_path "${TASK_JSON_GROUPS[i]}"
    done
done
