# # _except_mvtec_visa
# # _except_continual_ad

# json_path_list=(
#     "base_classes"
#     "base_classes_except_mvtec_visa"
#     "base_classes_except_continual_ad"
# )

# for ((i=0; i<${#json_path_list[@]}; i++)); do
#     json_path=${json_path_list[$i]}
#     echo "Running for json_path=$json_path"
#     python test.py \
#         --image_size 336 \
#         --batch_size 32 \
#         --json_path $json_path \
#         --task_id 0
# done


json_path_list=(
    "5classes_tasks"
    "5classes_tasks_except_mvtec_visa"
)

for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=12; continual_model_id>=1; continual_model_id--)); do
        for ((task_id=1; task_id<=continual_model_id; task_id++)); do
            echo "Running continual_model_id=$continual_model_id, task_id=$task_id"
            python test.py \
                --image_size 336 \
                --batch_size 32 \
                --json_path $json_path \
                --task_id $task_id \
                --continual_model_id $continual_model_id
        done
    done
done

for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=12; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --image_size 336 \
            --batch_size 32 \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done

###

json_path_list=(
    "5classes_tasks_except_continual_ad"
)


for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=6; continual_model_id>=1; continual_model_id--)); do
        for ((task_id=1; task_id<=continual_model_id; task_id++)); do
            echo "Running continual_model_id=$continual_model_id, task_id=$task_id"
            python test.py \
                --image_size 336 \
                --batch_size 32 \
                --json_path $json_path \
                --task_id $task_id \
                --continual_model_id $continual_model_id
        done
    done
done

for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=6; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --image_size 336 \
            --batch_size 32 \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done