#!/bin/bash

# 공통 하이퍼파라미터 설정
phase=$1  # base 또는 continual
data_dir="results/"
hf_path="vit_large_patch14_dinov2.lvd142m"
layers_to_extract_from="24"
image_size=336
batch_size=16
test_batch_size=16
lr=0.0005
lr_decay_factor=0.2
hidden_dim=2048
noise_std=0.25
num_fake_patches=-1
dsc_layers=1
dsc_heads=4
dsc_dropout=0.1
top_k=10
fake_feature_type="random"
smoothing_sigma=16
smoothing_radius=18
seed=0
log_every_n_steps=4
val_monitor="image_auroc"
log_pixel_metrics=1

wandb_project="continual-general-ad"

# Continual용 설정
base_epochs=50
inc_epochs=20
task_json_name="5classes_tasks"

# 실행
echo "[RUN] Phase: $phase"

python /workspace/MegaInspection/GeneralAD/main.py \
    --run_type general_ad \
    --phase $phase \
    --data_dir $data_dir \
    --hf_path $hf_path \
    --layers_to_extract_from $layers_to_extract_from \
    --image_size $image_size \
    --batch_size $batch_size \
    --test_batch_size $test_batch_size \
    --lr $lr \
    --lr_decay_factor $lr_decay_factor \
    --hidden_dim $hidden_dim \
    --noise_std $noise_std \
    --num_fake_patches $num_fake_patches \
    --dsc_layers $dsc_layers \
    --dsc_heads $dsc_heads \
    --dsc_dropout $dsc_dropout \
    --top_k $top_k \
    --fake_feature_type $fake_feature_type \
    --smoothing_sigma $smoothing_sigma \
    --smoothing_radius $smoothing_radius \
    --seed $seed \
    --log_every_n_steps $log_every_n_steps \
    --val_monitor $val_monitor \
    --log_pixel_metrics $log_pixel_metrics \
    --wandb_project $wandb_project \
    --base_epochs $base_epochs \
    --inc_epochs $inc_epochs \
    --task_json_name $task_json_name