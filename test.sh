#!/bin/bash
declare -a id_arr=("5.4" "6.7")

for model_id in "${id_arr[@]}"
do
    python test_mmdgan_transfer_edge.py --id $model_id --gpu_ids 1
done

python test_mmdgan_transfer_color.py --id 6.7 --gpu_ids 1
