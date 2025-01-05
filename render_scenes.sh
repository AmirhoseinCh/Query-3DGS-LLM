#!/bin/bash

models=(
    'Qwen2.5-0.5B-Instruct'
    'Qwen2.5-1.5B-Instruct'
    'Qwen2.5-3B-Instruct'
    'Qwen2.5-7B-Instruct'
)

# models=(
#     'qwen2.5-0.5B-canon_dataset'
#     'qwen2.5-1.5B-canon_dataset'
#     'qwen2.5-3B-canon_dataset'
#     'qwen2.5-7B-canon_dataset'
# )

# models=(
#     'llama-1B-Instruct'
#     'llama-3B-Instruct'
#     'llama-8B-Instruct'
#     'llama-1B-Instruct-canon_dataset'
#     'llama-3B-Instruct-canon_dataset'
#     'llama-8B-Instruct-canon_dataset'
# )


scenes=(
    "scene_012_30_forward"
    "scene_021_30_forward"
    # "scene_036_30_forward"
    # "scene_078_30_forward"
    # "scene_088_50_forward"
)
for model in "${models[@]}"; do
    for scene in "${scenes[@]}"; do
        python3 renderfromclip.py --source_path  "output/wayvescene/${scene}/1/rendered_clip/clip_features" --input_command  "Qwen/${model}_${scene}_indir.json"  --alpha  0.5  --scale 10  --com_type "argmax"
    done
done
# python3 renderfromclip.py --source_path  "output/wayvescene/${scene}/1/rendered_clip/clip_features" --input_command  "unsloth/Qwen/${model}_${scene}.json"  --alpha  0.5  --scale 10  --com_type "argmax"
# python3 renderfromclip.py --source_path  "output/wayvescene/${scene}/1/rendered_clip/clip_features" --input_command  "unsloth/Llama/${model}_${scene}_indir.json"  --alpha  0.5  --scale 10  --com_type "argmax"
# for scene in "${scenes[@]}"; do
#     python3 renderfromclip.py --config "expert/rele_configs/${scene}.cfg"
# done
# python3 renderfromclip.py --config "unsloth/Qwen/rele_configs/${scene}.cfg"
# python3 renderfromclip.py --config "expert/rele_configs/${scene}.cfg"