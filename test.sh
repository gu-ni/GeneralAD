# _except_mvtec_visa
# _except_continual_ad

json_path_list=(
    "base_classes"
    "base_classes_except_mvtec_visa"
    "base_classes_except_continual_ad"
)

for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    python test.py \
        --image_size 336 \
        --batch_size 16 \
        --json_path $json_path \
        --task_id 0
done