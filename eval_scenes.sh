#!/bin/bash

# models=(
#     'Qwen2.5-0.5B-Instruct'
#     'Qwen2.5-1.5B-Instruct'
#     'Qwen2.5-3B-Instruct'
#     'Qwen2.5-7B-Instruct'
# )
# models=(
#     'qwen2.5-0.5B-canon_dataset'
#     'qwen2.5-1.5B-canon_dataset'
#     'qwen2.5-3B-canon_dataset'
#     'qwen2.5-7B-canon_dataset'
# )

models=(
    'llama-1B-Instruct'
    'llama-3B-Instruct'
    'llama-8B-Instruct'
    'llama-1B-Instruct-canon_dataset'
    'llama-3B-Instruct-canon_dataset'
    'llama-8B-Instruct-canon_dataset'
)

scenes=(
    "scene_012_30_forward"
    "scene_021_30_forward"
    "scene_036_30_forward"
    "scene_078_30_forward"
    "scene_088_50_forward"
)

for model in "${models[@]}"; do
    for scene in "${scenes[@]}"; do
        python3 eval_mod.py -pr unsloth/Llama/${model}_${scene} -gt data/wayvescene/${scene}/output_segformer -js unsloth/Llama/${model}_${scene}.json
    done
done
# python3 eval_mod.py -pr unsloth/Qwen/${model}_${scene} -gt data/wayvescene/${scene}/output_segformer -js unsloth/Qwen/${model}_${scene}.json
# for scene in "${scenes[@]}"; do
#     python3 eval_mod.py -pr Neg/expert/${scene} -gt data/wayvescene/${scene}/output_segformer -js expert/${scene}.json
# done
# python3 eval_mod.py -pr unsloth/Qwen/qwen2.5-7B-canon_dataset_${scene} -gt data/wayvescene/${scene}/output_segformer -js unsloth/Qwen/qwen2.5-7B-canon_dataset_${scene}.json
# python3 eval_mod.py -pr base/expert/${scene} -gt data/wayvescene/${scene}/output_segformer -js expert/${scene}.json